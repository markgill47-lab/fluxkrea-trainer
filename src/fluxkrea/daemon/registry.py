"""Which dataset folders this node knows about.

Datasets are node-local: there is no shared mount and no coordinator, so
each node keeps its own list and the client assembles the fleet picture by
asking every node (doc 06, "who knows what lives where").

Ids are derived, not allocated. ``poses`` at ``/srv/data/poses`` gets the
id ``poses``, and only if two different folders want the same name does a
short path hash get appended. That means the same dataset has the same id
on every node it lives on, which is what makes ``fk dataset where poses``
and the drift report readable.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from ..core import paths
from ..core.dataset.naming import derive_id, path_digest, slug


@dataclass(frozen=True, slots=True)
class Dataset:
    id: str
    path: Path
    name: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "path": self.path.as_posix(),
            "name": self.name,
            "exists": self.path.is_dir(),
        }


class Registry:
    """The node's dataset list, persisted to ``data_dir/datasets.json``."""

    def __init__(self, file: Path | None = None) -> None:
        self._file = file if file is not None else paths.registry_file()
        self._lock = threading.Lock()
        self._entries: dict[str, Dataset] = {}
        self._load()

    # -- persistence ------------------------------------------------------

    def _load(self) -> None:
        if not self._file.is_file():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for entry in data.get("datasets", []):
            try:
                dataset = Dataset(
                    id=str(entry["id"]),
                    path=paths.expand(entry["path"]),
                    name=str(entry.get("name") or entry["id"]),
                )
            except (KeyError, TypeError):
                continue
            self._entries[dataset.id] = dataset

    def _save(self) -> None:
        paths.ensure_dir(self._file.parent)
        payload = {
            "version": 1,
            "datasets": [
                {"id": d.id, "path": d.path.as_posix(), "name": d.name}
                for d in sorted(self._entries.values(), key=lambda d: d.id)
            ],
        }
        tmp = self._file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._file)

    # -- access -----------------------------------------------------------

    def list(self) -> list[Dataset]:
        with self._lock:
            return sorted(self._entries.values(), key=lambda d: d.id)

    def get(self, dataset_id: str) -> Dataset | None:
        with self._lock:
            return self._entries.get(dataset_id)

    def by_path(self, path: str | os.PathLike[str]) -> Dataset | None:
        target = paths.resolve(path)
        with self._lock:
            for dataset in self._entries.values():
                try:
                    if dataset.path.resolve() == target:
                        return dataset
                except OSError:
                    continue
        return None

    def register(self, path: str | os.PathLike[str], name: str | None = None) -> Dataset:
        """Add a folder, or return the entry it already has.

        Registering is idempotent by path, so a client that re-registers on
        every push does not accumulate duplicates.
        """
        folder = paths.expand(path)
        existing = self.by_path(folder)
        if existing is not None:
            return existing

        with self._lock:
            base = slug(name) if name else derive_id(folder)
            dataset_id = base
            if dataset_id in self._entries:
                dataset_id = f"{base}-{path_digest(folder)}"
            dataset = Dataset(id=dataset_id, path=folder, name=name or folder.name)
            self._entries[dataset_id] = dataset
            self._save()
        return dataset

    def forget(self, dataset_id: str) -> bool:
        """Remove a registration. Never touches the folder itself."""
        with self._lock:
            if dataset_id not in self._entries:
                return False
            del self._entries[dataset_id]
            self._save()
        return True
