"""The fleet: client-side aggregation over a node list. No coordinator.

A coordinator would be a single point of failure and a second daemon to
deploy for a lab-sized fleet (doc 06). So the client holds the list, fans
out, and assembles the picture. Adding a node is a config line, and a node
being down degrades to one missing row rather than breaking the view.

    # ~/.fluxkrea/fleet.toml
    [[node]]
    name = "olympus-1"
    url  = "http://localhost:8471"   # via ssh -L
    [[node]]
    name = "olympus-2"
    url  = "http://localhost:8472"

Placement is explicit - ``fk dataset push --to olympus-2``. Automatic
scheduling is a later question, and a coordinator is the price of it.
"""

from __future__ import annotations

import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core import paths
from ..core.config import secret
from .client import ApiError, Client

#: Fan-out width. A lab fleet is small; this is about not opening fifty
#: sockets at once on a laptop, not about throughput.
FANOUT = 8

#: Per-node timeout when fanning out. A node that is down should cost a
#: second, not block the table.
PROBE_TIMEOUT = 5.0


@dataclass(frozen=True, slots=True)
class Node:
    name: str
    url: str
    #: Optional, and only used to build the rsync target for a push.
    ssh: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "url": self.url, "ssh": self.ssh}

    def client(self) -> Client:
        return Client.remote(self.url, secret("token"), timeout=PROBE_TIMEOUT)


@dataclass
class Fleet:
    nodes: list[Node] = field(default_factory=list)
    source: Path | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> Fleet:
        target = path or paths.fleet_file()
        if not target.is_file():
            return cls(nodes=[])
        try:
            data = tomllib.loads(target.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"cannot read {target}: {exc}") from exc

        nodes: list[Node] = []
        for entry in data.get("node", []):
            name = str(entry.get("name", "")).strip()
            url = str(entry.get("url", "")).strip()
            if not name or not url:
                continue
            nodes.append(Node(name=name, url=url.rstrip("/"), ssh=str(entry.get("ssh", ""))))
        return cls(nodes=nodes, source=target)

    def get(self, name: str) -> Node | None:
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def url_for(self, name: str) -> str:
        node = self.get(name)
        if node is None:
            known = ", ".join(n.name for n in self.nodes) or "none configured"
            raise ValueError(f"no node called {name!r} in {paths.fleet_file()}; known: {known}")
        return node.url

    # -- aggregation ------------------------------------------------------

    def _fanout(self, work: Any) -> list[Any]:
        if not self.nodes:
            return []
        with ThreadPoolExecutor(max_workers=min(FANOUT, len(self.nodes))) as pool:
            return list(pool.map(work, self.nodes))

    def status(self) -> list[dict[str, Any]]:
        """One row per node. A node that is down is a row saying so."""

        def probe(node: Node) -> dict[str, Any]:
            row: dict[str, Any] = {"node": node.name, "url": node.url, "state": "down"}
            client = node.client()
            try:
                health = client.get("/health")
                info = client.get("/node")
            except ApiError as exc:
                row["error"] = str(exc)
                return row
            finally:
                client.close()

            gpus = info.get("gpus") or []
            row.update(
                {
                    "state": "up",
                    "version": health.get("version"),
                    "queue_depth": health.get("queue_depth", 0),
                    "tasks_active": health.get("tasks_active", 0),
                    "torch": info.get("torch"),
                    "cuda": info.get("cuda"),
                    "driver": info.get("driver"),
                    "gpu": gpus[0]["name"] if gpus else None,
                    "gpus": gpus,
                }
            )
            return row

        return self._fanout(probe)

    def datasets(self) -> dict[str, list[dict[str, Any]]]:
        """Dataset placement across the fleet, with drift where it exists.

        Nobody knows centrally what lives where - that is the consequence of
        having no coordinator. The client asks each node and assembles it.
        """
        placement: dict[str, list[dict[str, Any]]] = {}

        def probe(node: Node) -> tuple[str, list[dict[str, Any]]]:
            client = node.client()
            rows: list[dict[str, Any]] = []
            try:
                for dataset in client.get("/datasets").get("datasets", []):
                    row = {
                        "node": node.name,
                        "dataset": dataset["id"],
                        "path": dataset["path"],
                        "state": "ok" if dataset["exists"] else "missing",
                    }
                    try:
                        summary = client.get(
                            f"/datasets/{dataset['id']}/manifest",
                            params={"digests": False},
                            timeout=60.0,
                        )
                        row["files"] = summary["files"]
                        row["bytes"] = summary["bytes"]
                        row["fingerprint"] = _fingerprint(summary)
                    except ApiError:
                        row["state"] = "unreadable"
                    rows.append(row)
            except ApiError as exc:
                rows.append({"node": node.name, "state": "down", "error": str(exc)})
            finally:
                client.close()
            return node.name, rows

        for _, rows in self._fanout(probe):
            for row in rows:
                dataset_id = row.get("dataset")
                if dataset_id:
                    placement.setdefault(dataset_id, []).append(row)

        _mark_drift(placement)
        return placement

    def where(self, dataset_id: str) -> list[dict[str, Any]]:
        """The nodes that have one dataset, flagging any that disagree."""
        return self.datasets().get(dataset_id, [])

    @staticmethod
    def example() -> str:
        return (
            "\n"
            "  [[node]]\n"
            '  name = "olympus-1"\n'
            '  url  = "http://localhost:8471"   # via ssh -N -L 8471:localhost:8471\n'
            "\n"
            "  [[node]]\n"
            '  name = "olympus-2"\n'
            '  url  = "http://localhost:8472"\n'
        )


def _fingerprint(summary: dict[str, Any]) -> str:
    """A cheap comparable summary of a copy: file count and total bytes.

    Deliberately not a digest of digests - the manifest is fetched without
    them here so that ``fk fleet datasets`` stays fast over a whole fleet.
    Two copies that differ in content but match in size and count will read
    as agreeing; ``fk dataset push --dry-run`` is the exact answer.
    """
    return f"{summary.get('files', 0)}:{summary.get('bytes', 0)}"


def _mark_drift(placement: dict[str, list[dict[str, Any]]]) -> None:
    """Flag datasets whose copies disagree.

    Detecting drift is in scope; reconciling it is not. The client reports
    and the human decides which copy wins (doc 06).
    """
    for rows in placement.values():
        prints = {row.get("fingerprint") for row in rows if row.get("fingerprint")}
        if len(prints) > 1:
            for row in rows:
                if row.get("state") == "ok":
                    row["state"] = "drift"
