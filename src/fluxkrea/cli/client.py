"""The HTTP client every command goes through.

Doc 02: clients never import ``core`` directly, even when running on the
same machine - they go through the API, so the local case can never
quietly diverge from the remote one. This is the module that makes that
true; the CLI holds no dataset logic at all.

Two transports, one code path:

* **remote** - an ordinary HTTP client, pointed at a node. Reaching a lab
  box is an SSH tunnel and a different base URL; nothing about a command
  changes.
* **embedded** - the same daemon, started on an ephemeral loopback port
  for the duration of one command. This is what makes ``fk dataset scan
  ./poses`` work on a desktop with no service running, *without* growing a
  second code path: the request crosses a real socket into the same
  routers, the same serialisation and the same task runner. See
  ``embedded.py``.

The embedded case is chosen only when nothing is already listening
locally, so the CLI and a running daemon never fight over one dataset
folder. It costs about a second of startup per command, which is the price
of not having a divergent local path; ``fk serve`` avoids it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from ..core import paths
from ..core.config import Config, secret

DEFAULT_TIMEOUT = 30.0

#: Long enough for a manifest with digests over a large dataset, which is
#: bounded by disk read speed rather than anything the daemon controls.
SLOW_TIMEOUT = 600.0

#: An upload of a whole dataset. Generous, because the alternative is a
#: timeout part way through a multi-gigabyte push.
UPLOAD_TIMEOUT = 3600.0

API = "/api/v1"


class ApiError(Exception):
    """The daemon said no, or could not be reached."""

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


class Client:
    """One node's API."""

    def __init__(
        self,
        http: httpx.Client,
        *,
        base_url: str,
        token: str | None = None,
        embedded: bool = False,
    ) -> None:
        self._http = http
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.embedded = embedded
        self._daemon: Any = None

    # -- construction -----------------------------------------------------

    @classmethod
    def remote(cls, url: str, token: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> Client:
        base = url.rstrip("/")
        headers = {"x-fluxkrea-token": token} if token else {}
        return cls(
            httpx.Client(base_url=base, headers=headers, timeout=timeout),
            base_url=base,
            token=token,
        )

    @classmethod
    def local(cls, config: Config) -> Client:
        """Start a daemon on an ephemeral loopback port, just for this command."""
        from .embedded import EmbeddedDaemon

        daemon = EmbeddedDaemon(config)
        url = daemon.start()
        client = cls.remote(url, secret("token"))
        client.embedded = True
        client._daemon = daemon
        return client

    @classmethod
    def for_config(cls, config: Config, url: str | None = None, *, force_local: bool = False) -> Client:
        """Pick a transport: an explicit URL, a running daemon, or embedded."""
        explicit = url or os.environ.get("FLUXKREA_URL")
        if explicit:
            return cls.remote(explicit, secret("token"))
        if force_local:
            return cls.local(config)

        candidate = cls.remote(
            f"http://{_host(config.daemon.host)}:{config.daemon.port}", secret("token"), timeout=2.0
        )
        if candidate.reachable():
            candidate._http.timeout = httpx.Timeout(DEFAULT_TIMEOUT)
            return candidate
        candidate.close()
        return cls.local(config)

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:  # noqa: BLE001 - closing must never be the failure
            pass
        if self._daemon is not None:
            self._daemon.stop()
            self._daemon = None

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def where(self) -> str:
        return "this machine" if self.embedded else self.base_url

    # -- plumbing ---------------------------------------------------------

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: dict[str, Any] | None = None,
        content: Any = None,
        timeout: float | None = None,
    ) -> Any:
        try:
            response = self._http.request(
                method,
                f"{API}{path}",
                json=json_body,
                params=_clean(params),
                content=content,
                timeout=timeout or DEFAULT_TIMEOUT,
            )
        except httpx.RequestError as exc:
            raise ApiError(_unreachable(self, exc)) from exc

        if response.status_code >= 400:
            raise ApiError(_message(response), response.status_code)
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Any:
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> Any:
        return self.request("PUT", path, **kwargs)

    def patch(self, path: str, **kwargs: Any) -> Any:
        return self.request("PATCH", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Any:
        return self.request("DELETE", path, **kwargs)

    def download(self, path: str, target: Path, *, params: dict[str, Any] | None = None) -> Path:
        """Stream a response to a file, without holding it in memory."""
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._http.stream(
                "GET", f"{API}{path}", params=_clean(params), timeout=SLOW_TIMEOUT
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise ApiError(_message(response), response.status_code)
                with target.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
        except httpx.RequestError as exc:
            raise ApiError(_unreachable(self, exc)) from exc
        return target

    # -- events -----------------------------------------------------------

    def stream(self, path: str, since: int = -1) -> Iterator[dict[str, Any]]:
        """Follow an SSE stream, yielding decoded event payloads.

        Plain line reading rather than an SSE library: the wire format is
        four line shapes, and one fewer client dependency is worth more
        than the abstraction.
        """
        params = {"since": since} if since >= 0 else None
        try:
            with self._http.stream(
                "GET",
                f"{API}{path}",
                params=_clean(params),
                headers={"accept": "text/event-stream"},
                timeout=httpx.Timeout(SLOW_TIMEOUT, read=None),
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise ApiError(_message(response), response.status_code)
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        try:
                            yield json.loads(line[5:].strip())
                        except ValueError:
                            continue
        except httpx.RequestError as exc:
            raise ApiError(_unreachable(self, exc)) from exc

    # -- convenience ------------------------------------------------------

    def reachable(self) -> bool:
        try:
            self.get("/health", timeout=3.0)
        except ApiError:
            return False
        return True

    def register(
        self,
        path: str | os.PathLike[str],
        name: str | None = None,
        *,
        create: bool = False,
    ) -> dict[str, Any]:
        """Register a folder and return the dataset record.

        Idempotent on the daemon side, so a command can call this rather
        than making somebody register before doing anything. ``create``
        makes the folder if it is missing - which a first push to a fresh
        node needs, and nothing else should use.
        """
        return self.post(
            "/datasets",
            json_body={"path": paths.expand(path).as_posix(), "name": name, "create": create},
        )

    def resolve(self, target: str) -> dict[str, Any]:
        """Accept a dataset id or a folder path, and return the record.

        A path resolves against *this node's* filesystem, which is the
        correct asymmetry: pushing to a remote node names a local folder
        and a remote id, and conflating the two is how a client ends up
        asking a node about a path that only exists on the laptop.
        """
        listing = self.get("/datasets").get("datasets", [])
        for dataset in listing:
            if dataset["id"] == target:
                return dataset

        candidate = paths.expand(target)
        wanted = candidate.as_posix().rstrip("/").lower()
        for dataset in listing:
            if str(dataset["path"]).rstrip("/").lower() == wanted:
                return dataset

        if candidate.is_dir():
            return self.register(candidate)
        raise ApiError(f"no dataset {target!r} on {self.where}, and no such folder here", 404)

    def watch(self, task_id: str, *, kind: str = "tasks") -> dict[str, Any]:
        """Follow a task or job to completion, yielding nothing but the end.

        Used by commands that submit work and then wait, which is most of
        them - the CLI is synchronous even though the API is not.
        """
        for payload in self.stream(f"/{kind}/{task_id}/events"):
            if payload.get("kind") == "finished":
                break
        return self.get(f"/{kind}/{task_id}")


def _host(host: str) -> str:
    """A daemon bound to 0.0.0.0 is still reached at localhost from here."""
    return "127.0.0.1" if host in ("0.0.0.0", "::", "") else host  # noqa: S104


def _clean(params: dict[str, Any] | None) -> dict[str, Any] | None:
    if not params:
        return None
    return {k: v for k, v in params.items() if v is not None}


def _unreachable(client: Client, exc: Exception) -> str:
    if client.embedded:
        return f"the embedded daemon failed: {exc}"
    return (
        f"cannot reach {client.base_url}: {exc}. "
        "Is the daemon running there, and is the SSH tunnel up?"
    )


def _message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"{response.status_code}: {response.text.strip()[:400]}"
    if isinstance(payload, dict):
        if "error" in payload:
            return str(payload["error"])
        if "detail" in payload:
            detail = payload["detail"]
            if isinstance(detail, list) and detail:
                first = detail[0]
                where = ".".join(str(p) for p in first.get("loc", [])[1:])
                message = str(first.get("msg", "invalid"))
                return f"{where}: {message}" if where else message
            return str(detail)
    return f"{response.status_code}"
