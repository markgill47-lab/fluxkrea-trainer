"""``fk dataset push`` - moving dataset bytes to a node.

**Control goes over the API; bulk data does not.** Reimplementing rsync
over HTTP would be a poor use of effort when every node is already
reachable over SSH (doc 06, "moving bytes"). So:

1. Ask the target for its manifest.
2. Diff against the local one - only differing files are candidates.
3. Transfer by the best transport available:
   * **rsync over SSH** where present. Incremental, compressed, resumable.
   * **tar into ``POST /datasets/{id}/import``** otherwise. Portable, works
     from a Windows laptop with nothing installed, one request.
4. Rescan on the target.

Windows is the reason the fallback has to be good: OpenSSH ships with
Windows 10+ and rsync does not.

``--sidecars-only`` is the one that makes iterating practical. Images are
large and static; captions and masks are small and change constantly. Once
the images are on a node, a re-masked pass moves kilobytes.
"""

from __future__ import annotations

import io
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core import paths
from ..core.dataset import manifest
from ..core.dataset.archive import stream as tar_stream
from ..core.events import Emitter, Log, Progress, no_op, safe
from .client import ApiError, Client

RSYNC = "rsync"
TAR = "tar"


@dataclass
class PushResult:
    dataset_id: str
    node: str
    transport: str = ""
    files: int = 0
    bytes: int = 0
    dry_run: bool = False
    diff: Any = None
    error: str = ""
    remote: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error

    def summary(self) -> str:
        if self.error:
            return f"push failed: {self.error}"
        if self.diff is not None and self.diff.in_sync:
            return f"{self.dataset_id} is already in sync with {self.node}"
        verb = "would send" if self.dry_run else f"sent via {self.transport}"
        return f"{verb} {self.files} files ({manifest.human(self.bytes)}) to {self.node}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_id,
            "node": self.node,
            "transport": self.transport,
            "files": self.files,
            "bytes": self.bytes,
            "dry_run": self.dry_run,
            "ok": self.ok,
            "error": self.error,
            "diff": self.diff.as_dict() if self.diff is not None else None,
            "remote": self.remote,
        }


def push(
    client: Client,
    local: str | Path,
    *,
    dataset_id: str | None = None,
    sidecars_only: bool = False,
    dry_run: bool = False,
    transport: str = "auto",
    ssh_target: str | None = None,
    remote_path: str | None = None,
    digests: bool = True,
    emit: Emitter = no_op,
) -> PushResult:
    """Send a local dataset folder to the node *client* points at."""
    emit = safe(emit)
    source = paths.expand(local)
    if not source.is_dir():
        raise NotADirectoryError(f"not a dataset folder: {source}")

    remote = _remote_dataset(client, source, dataset_id, remote_path)
    result = PushResult(dataset_id=remote["id"], node=client.base_url, dry_run=dry_run, remote=remote)

    emit(Log(line=f"Comparing with {remote['id']} on {client.base_url}"))
    local_manifest = manifest.build(
        source, digests=digests, sidecars_only=sidecars_only
    )
    theirs = manifest.Manifest.from_dict(
        client.get(
            f"/datasets/{remote['id']}/manifest",
            params={"digests": digests, "sidecars_only": sidecars_only},
            timeout=600.0,
        )
    )

    diff = local_manifest.diff(theirs)
    result.diff = diff
    result.files = len(diff.transfers)
    result.bytes = diff.bytes
    emit(Log(line=diff.summary()))

    if diff.in_sync or dry_run or not diff.transfers:
        result.transport = "none"
        return result

    chosen = _choose(transport, ssh_target)
    result.transport = chosen
    emit(Log(line=f"Transferring by {chosen}"))

    try:
        if chosen == RSYNC:
            _rsync(source, ssh_target or "", remote["path"], diff, emit)
        else:
            _tar(client, remote["id"], source, diff, emit)
    except (OSError, ApiError, subprocess.SubprocessError) as exc:
        result.error = str(exc)
        emit(Log(line=result.error, level="error"))
        return result

    emit(Log(line="Rescanning on the node"))
    try:
        client.post(f"/datasets/{remote['id']}/scan", timeout=600.0)
    except ApiError as exc:
        emit(Log(line=f"transferred, but the rescan failed: {exc}", level="warning"))

    return result


# --------------------------------------------------------------------------
# transports
# --------------------------------------------------------------------------


def _choose(requested: str, ssh_target: str | None) -> str:
    if requested == RSYNC:
        if not ssh_target:
            raise ValueError("rsync needs --ssh user@host")
        if not shutil.which(RSYNC):
            raise ValueError("rsync is not on PATH")
        return RSYNC
    if requested == TAR:
        return TAR
    # auto: rsync when it can actually work, tar otherwise. On Windows that
    # is almost always tar, which is exactly why the tar path must be good.
    if ssh_target and shutil.which(RSYNC):
        return RSYNC
    return TAR


def _rsync(source: Path, ssh_target: str, remote_path: str, diff: Any, emit: Emitter) -> None:
    """Hand the file list to rsync and let it do what it is good at."""
    listing = "\n".join(entry.path for entry in diff.transfers) + "\n"
    destination = f"{ssh_target}:{remote_path.rstrip('/')}/"

    command = [
        RSYNC,
        "-az",
        "--partial",
        "--files-from=-",
        f"{source.as_posix().rstrip('/')}/",
        destination,
    ]
    emit(Log(line=" ".join(shlex.quote(part) for part in command), level="debug"))

    process = subprocess.run(  # noqa: S603 - argument list, no shell
        command,
        input=listing,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise OSError(f"rsync exited {process.returncode}: {process.stderr.strip()[:400]}")


def _tar(client: Client, dataset_id: str, source: Path, diff: Any, emit: Emitter) -> None:
    """One request, one tar. The transport that needs nothing installed."""
    members = [entry.path for entry in diff.transfers]
    total = len(members)
    emit(Progress(step=0, total=total, message="Packing"))

    buffer = io.BytesIO()
    for chunk in tar_stream(source, members):
        buffer.write(chunk)
    emit(Progress(step=total, total=total, message="Packing"))

    payload = buffer.getvalue()
    emit(Log(line=f"Uploading {manifest.human(len(payload))}"))
    from .client import UPLOAD_TIMEOUT

    client.post(f"/datasets/{dataset_id}/import", content=payload, timeout=UPLOAD_TIMEOUT)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _remote_dataset(
    client: Client,
    source: Path,
    dataset_id: str | None,
    remote_path: str | None,
) -> dict[str, Any]:
    """Find or create the dataset on the target node.

    A path that exists locally means nothing on the far side, so the remote
    folder is either named explicitly or assumed to share the local folder
    name under one of the node's roots.
    """
    wanted = dataset_id or manifest_id(source)

    for dataset in client.get("/datasets").get("datasets", []):
        if dataset["id"] == wanted:
            return dataset

    if remote_path:
        # Naming a remote path is how you place a dataset a node has never
        # had, so the folder is expected not to exist yet.
        return client.register(remote_path, wanted, create=True)

    roots = client.get("/config").get("dataset", {}).get("roots") or []
    if not roots:
        raise ApiError(
            f"{client.base_url} has no dataset called {wanted!r} and no configured roots, "
            "so there is nowhere obvious to put it. Pass --remote-path.",
            404,
        )
    # The folder will not exist on a node that has never seen this dataset,
    # so ask the daemon to create it inside its own root rather than
    # requiring somebody to mkdir over SSH first.
    return client.register(
        f"{str(roots[0]).rstrip('/')}/{source.name}", wanted, create=True
    )


def manifest_id(source: Path) -> str:
    from ..core.dataset.naming import derive_id

    return derive_id(source)
