"""Projects - a named group of dataset folders sharing one training config.

A teaching room does not have one dataset, it has one per student and
several per student by the end of the afternoon. Without a grouping the
dataset selector is a flat list of forty folders with no clue which three
belong together, and the training form is re-filled from scratch every
time somebody switches between them.

So a project holds three things and deliberately not a fourth:

* a **name**, which is the only identity in the system. There are no
  accounts and no passwords; the project a browser has open is who the
  queue thinks you are.
* a **list of dataset ids**, not paths. Datasets stay registered on the
  node exactly as they were - this is a view over :mod:`registry`, not a
  replacement for it, so ``fk dataset register`` keeps working and a
  folder can appear in two projects without being copied.
* a **shared config**: the training form's state, held here rather than in
  one student's browser so that the three datasets in a project train with
  the same settings and the second student to sit down inherits them.

The fourth thing it does not hold is any of the datasets' *contents*.
Deleting a project deletes a grouping and nothing on disk, which is the
same promise ``forget`` makes about a dataset.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core import paths
from ..core.dataset.naming import slug

#: A project must be nameable by a person in a hurry, and the name is what
#: appears beside their job in a shared queue. Long enough to be specific,
#: short enough to read in a list.
MAX_NAME = 64


@dataclass
class Project:
    """One project. ``config`` is opaque here on purpose - see below."""

    id: str
    name: str
    datasets: list[str] = field(default_factory=list)
    #: The training form, verbatim. Kept opaque because the shape belongs
    #: to the client that renders it: pinning a schema here would mean a
    #: daemon release every time the form grows a field, and the daemon has
    #: no opinion about any of them. What it *is* is validated on submit,
    #: by ``RunSpec.from_dict``, which is the place that has to care.
    config: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "datasets": list(self.datasets),
            "config": dict(self.config),
            "created": self.created,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Project:
        datasets = data.get("datasets") or []
        config = data.get("config") or {}
        return cls(
            id=str(data["id"]),
            name=str(data.get("name") or data["id"]),
            datasets=[str(d) for d in datasets if str(d).strip()],
            config=dict(config) if isinstance(config, dict) else {},
            created=float(data.get("created", time.time())),
            updated=float(data.get("updated", time.time())),
        )


class ProjectError(Exception):
    """A project operation that will not be performed."""


def check_name(name: str) -> str:
    """Validate a project name, returning the trimmed version."""
    text = str(name or "").strip()
    if not text:
        raise ProjectError("a project needs a name")
    if len(text) > MAX_NAME:
        raise ProjectError(f"a project name is at most {MAX_NAME} characters")
    return text


class ProjectStore:
    """The node's projects, persisted to ``data_dir/projects.json``.

    Same shape as :class:`registry.Registry` deliberately: one lock, one
    atomic write, ids derived from the name rather than allocated. A room
    of students all pressing "New project" at once is the normal case here
    rather than the exotic one, so every mutation takes the lock and every
    write goes through a temp file - two browsers saving a config at the
    same moment must not leave half a file behind.
    """

    def __init__(self, file: Path | None = None) -> None:
        self._file = file if file is not None else paths.projects_file()
        self._lock = threading.RLock()
        self._entries: dict[str, Project] = {}
        self._load()

    # -- persistence ------------------------------------------------------

    def _load(self) -> None:
        if not self._file.is_file():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for entry in data.get("projects", []):
            try:
                project = Project.from_dict(entry)
            except (KeyError, TypeError, ValueError):
                continue  # one malformed project must not lose the rest
            self._entries[project.id] = project

    def _save(self) -> None:
        paths.ensure_dir(self._file.parent)
        payload = {
            "version": 1,
            "projects": [
                project.as_dict()
                for project in sorted(self._entries.values(), key=lambda p: p.created)
            ],
        }
        tmp = self._file.with_suffix(f".json.{id(self)}.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._file)
        finally:
            tmp.unlink(missing_ok=True)

    # -- access -----------------------------------------------------------

    def list(self) -> list[Project]:
        with self._lock:
            return sorted(self._entries.values(), key=lambda p: p.created)

    def get(self, project_id: str) -> Project | None:
        with self._lock:
            return self._entries.get(project_id)

    def require(self, project_id: str) -> Project:
        found = self.get(project_id)
        if found is None:
            raise ProjectError(f"no project {project_id!r}")
        return found

    def __contains__(self, project_id: object) -> bool:
        with self._lock:
            return project_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    # -- mutation ---------------------------------------------------------

    def create(self, name: str, datasets: list[str] | None = None) -> Project:
        """Make a project. Two projects may share a name but never an id."""
        clean = check_name(name)
        with self._lock:
            # `slug` falls back to the literal "dataset" for a name with no
            # slug-able characters in it at all, which is the wrong noun
            # here and would read as a bug in a project list.
            base = slug(clean)
            if base == "dataset" and "dataset" not in clean.lower():
                base = "project"
            project_id = base
            suffix = 2
            while project_id in self._entries:
                project_id = f"{base}-{suffix}"
                suffix += 1
            project = Project(
                id=project_id,
                name=clean,
                datasets=list(datasets or []),
            )
            self._entries[project_id] = project
            self._save()
        return project

    def rename(self, project_id: str, name: str) -> Project:
        """Change the display name. **The id does not move.**

        Renaming is expected to be common - a project is created before
        anyone knows what it is for - and the id is what jobs, the queue
        and every open browser hold. Deriving a new id from the new name
        would orphan all of them silently, which is exactly the class of
        bug this file is arranged to avoid.
        """
        clean = check_name(name)
        with self._lock:
            project = self.require(project_id)
            project.name = clean
            project.updated = time.time()
            self._save()
            return project

    def set_config(self, project_id: str, config: dict[str, Any]) -> Project:
        """Replace the shared config wholesale.

        Replace rather than merge: the client sends the whole form, and a
        merge would make "I cleared that field" indistinguishable from "I
        did not mention it" - the same reason ``PUT /config`` is flat.
        """
        with self._lock:
            project = self.require(project_id)
            project.config = dict(config)
            project.updated = time.time()
            self._save()
            return project

    def add_dataset(self, project_id: str, dataset_id: str) -> Project:
        with self._lock:
            project = self.require(project_id)
            if dataset_id not in project.datasets:
                project.datasets.append(dataset_id)
                project.updated = time.time()
                self._save()
            return project

    def remove_dataset(self, project_id: str, dataset_id: str) -> Project:
        """Drop a dataset from a project. The folder is untouched."""
        with self._lock:
            project = self.require(project_id)
            if dataset_id in project.datasets:
                project.datasets.remove(dataset_id)
                project.updated = time.time()
                self._save()
            return project

    def delete(self, project_id: str) -> bool:
        """Forget a project. Its datasets stay registered on the node."""
        with self._lock:
            if project_id not in self._entries:
                return False
            del self._entries[project_id]
            self._save()
        return True

    def prune(self, known: set[str]) -> int:
        """Drop references to datasets this node no longer has registered.

        A student deregisters a folder and every project still listing it
        would otherwise show a dataset that 404s on click.
        """
        removed = 0
        with self._lock:
            for project in self._entries.values():
                kept = [d for d in project.datasets if d in known]
                if len(kept) != len(project.datasets):
                    removed += len(project.datasets) - len(kept)
                    project.datasets = kept
                    project.updated = time.time()
            if removed:
                self._save()
        return removed


__all__ = ["MAX_NAME", "Project", "ProjectError", "ProjectStore", "check_name"]
