"""Persisted face boxes and review state - ``face_boxes.json`` (doc 04).

    {
      "punch_014.jpg": {
        "boxes": [{"x": 412, "y": 88, "w": 96, "h": 128, "src": "yunet", "conf": 0.91},
                  {"x": 640, "y": 120, "w": 88, "h": 110, "src": "manual"}],
        "reviewed": true
      }
    }

Why this file exists at all: detection is the expensive step and review is
the slow one, and neither should ever have to be repeated. Regenerating
masks from stored boxes is instant, so changing the expansion factor
re-renders every mask without re-detecting, and a review pass survives any
number of re-exports.

It is also small - kilobytes for a few hundred images - which is what
makes ``fk dataset push --sidecars-only`` a practical loop across the
fleet (doc 06).
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import paths
from ..detect.base import MANUAL, Box


@dataclass(slots=True)
class ImageBoxes:
    """One image's boxes and whether a human has signed them off."""

    boxes: list[Box] = field(default_factory=list)
    reviewed: bool = False

    @property
    def empty(self) -> bool:
        return not self.boxes

    @property
    def manual(self) -> list[Box]:
        return [b for b in self.boxes if b.manual]

    @property
    def detected(self) -> list[Box]:
        return [b for b in self.boxes if not b.manual]

    def as_dict(self) -> dict[str, Any]:
        return {"boxes": [b.as_dict() for b in self.boxes], "reviewed": self.reviewed}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageBoxes:
        raw = data.get("boxes") or []
        boxes: list[Box] = []
        for entry in raw:
            if isinstance(entry, dict):
                try:
                    boxes.append(Box.from_dict(entry))
                except ValueError:
                    continue  # one malformed box must not lose the rest
        return cls(boxes=boxes, reviewed=bool(data.get("reviewed", False)))


#: One lock per box file, so two writers to the same dataset queue rather
#: than race. Keyed by resolved path: two datasets are independent.
_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

#: Makes each writer's temp file its own, even within one process.
_TEMP_SEQUENCE = itertools.count()

#: Windows hands out a sharing violation when an indexer or a scanner has
#: the file open for a moment. Worth a few retries before giving up on
#: somebody's review pass.
REPLACE_ATTEMPTS = 5
REPLACE_BACKOFF = 0.05


def _lock_for(path: Path) -> threading.Lock:
    key = path.resolve() if path.parent.exists() else path
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.Lock()
        return lock


def _replace_atomically(tmp: Path, target: Path) -> None:
    """``tmp`` becomes ``target``, retrying a transient Windows refusal."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            tmp.replace(target)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_BACKOFF * (attempt + 1))


@dataclass
class BoxStore:
    """The whole sidecar file, keyed by image filename."""

    root: Path
    entries: dict[str, ImageBoxes] = field(default_factory=dict)

    # -- lifecycle --------------------------------------------------------

    @classmethod
    def load(cls, root: str | os.PathLike[str]) -> BoxStore:
        folder = paths.expand(root)
        target = paths.boxes_file(folder)
        if not target.is_file():
            return cls(root=folder)
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Unlike metadata, this file holds human review work. Losing it
            # silently would throw away an afternoon, so say so.
            raise ValueError(f"{target} is not readable as JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"{target} is not a box file")

        entries = data.get("items", data)  # accept the bare doc-04 shape too
        store = cls(root=folder)
        for name, payload in entries.items():
            if isinstance(payload, dict):
                store.entries[str(name)] = ImageBoxes.from_dict(payload)
        return store

    def save(self) -> Path:
        target = paths.boxes_file(self.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: entry.as_dict() for name, entry in sorted(self.entries.items())}

        # A temp name of this writer's own. It used to be a fixed
        # `.json.tmp`, so two review marks arriving together wrote the same
        # file and one renamed it out from under the other - on Windows
        # that is `PermissionError: [WinError 5] Access is denied`, a 500,
        # and a review screen that stops responding.
        tmp = target.with_name(f"{target.name}.{os.getpid()}.{next(_TEMP_SEQUENCE)}.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            _replace_atomically(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        return target

    @classmethod
    def update(cls, root: str | os.PathLike[str], mutate: Callable[[BoxStore], Any]) -> BoxStore:
        """Load, change and save under one lock.

        Read-modify-write was not atomic: two marks arriving together both
        loaded the file, both changed their own copy, and the second save
        wrote the first one's change away. Silent, and this file holds
        human review work - the thing the module docstring says losing
        would throw away an afternoon.
        """
        folder = paths.expand(root)
        with _lock_for(paths.boxes_file(folder)):
            store = cls.load(folder)
            mutate(store)
            store.save()
            return store

    # -- access -----------------------------------------------------------

    def get(self, filename: str) -> ImageBoxes:
        return self.entries.get(filename, ImageBoxes())

    def boxes(self, filename: str) -> list[Box]:
        return list(self.get(filename).boxes)

    def is_reviewed(self, filename: str) -> bool:
        return self.get(filename).reviewed

    def set_boxes(self, filename: str, boxes: Iterable[Box], *, reviewed: bool | None = None) -> None:
        """Replace an image's boxes. What ``PUT .../boxes`` calls."""
        entry = self.entries.setdefault(filename, ImageBoxes())
        entry.boxes = list(boxes)
        if reviewed is not None:
            entry.reviewed = reviewed

    def record_detection(self, filename: str, boxes: Iterable[Box]) -> None:
        """Store a detector's output, **keeping every manual box**.

        Re-detecting with a different threshold must never discard boxes a
        human drew - those are the ones covering the faces the detector
        could not find, which is the whole reason review exists. Recording
        new detections also clears the reviewed flag, because there is now
        something the reviewer has not seen.
        """
        entry = self.entries.setdefault(filename, ImageBoxes())
        manual = entry.manual
        entry.boxes = [*boxes, *manual]
        entry.reviewed = False

    def mark_reviewed(self, filename: str, reviewed: bool = True) -> None:
        self.entries.setdefault(filename, ImageBoxes()).reviewed = reviewed

    def add_box(self, filename: str, box: Box) -> None:
        self.entries.setdefault(filename, ImageBoxes()).boxes.append(box)

    def clear(self, filename: str) -> None:
        self.entries.pop(filename, None)

    def __contains__(self, filename: object) -> bool:
        return filename in self.entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    # -- following operations ---------------------------------------------

    def apply_rename(self, mapping: dict[str, str]) -> None:
        self.entries = {mapping.get(name, name): entry for name, entry in self.entries.items()}

    def prune(self, keep: set[str]) -> int:
        stale = [name for name in self.entries if name not in keep]
        for name in stale:
            del self.entries[name]
        return len(stale)

    # -- review progress ---------------------------------------------------

    def progress(self, filenames: Iterable[str]) -> ReviewProgress:
        """``184/210 reviewed, 6 with no detections`` - doc 04's readout."""
        names = list(filenames)
        reviewed = sum(1 for n in names if self.is_reviewed(n))
        empty = [n for n in names if self.get(n).empty]
        unseen = [n for n in names if n not in self.entries]
        return ReviewProgress(
            total=len(names),
            reviewed=reviewed,
            empty=empty,
            undetected=unseen,
        )


@dataclass(frozen=True, slots=True)
class ReviewProgress:
    total: int
    reviewed: int
    #: Images with no boxes at all. **These are where misses hide** - doc 04
    #: wants them flagged loudly and sortable to the front of the review.
    empty: list[str]
    #: Images detection has never run against.
    undetected: list[str]

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.reviewed == self.total

    def summary(self) -> str:
        text = f"{self.reviewed}/{self.total} reviewed"
        if self.empty:
            text += f", {len(self.empty)} with no detections"
        if self.undetected:
            text += f", {len(self.undetected)} never detected"
        return text

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "reviewed": self.reviewed,
            "complete": self.complete,
            "empty": self.empty,
            "undetected": self.undetected,
            "summary": self.summary(),
        }


__all__ = ["MANUAL", "Box", "BoxStore", "ImageBoxes", "ReviewProgress"]
