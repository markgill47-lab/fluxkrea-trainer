"""Derived metadata for a dataset - a cache, never a source of truth.

v1 stores captions in ``.txt`` sidecars *and* in a JSON file, resolving
precedence case by case. Two sources of truth that drift (doc 01).

v2: ``.txt`` sidecars are the caption. This file holds only what is
derived and rebuildable - quality ratings, review state, cached image
dimensions - keyed by image filename. Delete it and nothing training-
relevant is lost.

Face boxes get their own file (``face_boxes.json``), because they are
edited by a review pass that has nothing to do with captioning and
everything to do with masking. See doc 04.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import paths

QUALITY_VALUES = ("good", "ok", "bad")


@dataclass
class Metadata:
    """The ``metadata.json`` beside a dataset. Load, mutate, save."""

    root: Path
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    # -- lifecycle --------------------------------------------------------

    @classmethod
    def load(cls, root: str | Path) -> Metadata:
        """Read ``metadata.json``, falling back to v1's descriptions file.

        A corrupt or unreadable file yields empty metadata rather than an
        exception: this is a cache, and refusing to open a dataset because
        a derived file got truncated would be the wrong trade.
        """
        root = paths.expand(root)
        target = paths.metadata_file(root)
        if target.is_file():
            try:
                data = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return cls(root=root)
            entries = data.get("items", {}) if isinstance(data, dict) else {}
            return cls(root=root, entries={str(k): dict(v) for k, v in entries.items()})
        return cls(root=root, entries=_read_v1_descriptions(root))

    def save(self) -> Path:
        """Write atomically. Nothing here is precious, but half a file is noise."""
        target = paths.metadata_file(self.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "items": {k: v for k, v in sorted(self.entries.items()) if v}}
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
        return target

    # -- access -----------------------------------------------------------

    def get(self, filename: str) -> dict[str, Any]:
        return self.entries.get(filename, {})

    def quality(self, filename: str) -> str | None:
        value = self.get(filename).get("quality")
        return value if value in QUALITY_VALUES else None

    def set_quality(self, filename: str, quality: str | None) -> None:
        if quality is not None and quality not in QUALITY_VALUES:
            raise ValueError(f"quality must be one of {QUALITY_VALUES}, got {quality!r}")
        self._entry(filename)["quality"] = quality
        if quality is None:
            self._entry(filename).pop("quality", None)

    def set(self, filename: str, key: str, value: Any) -> None:
        if value is None:
            self._entry(filename).pop(key, None)
        else:
            self._entry(filename)[key] = value

    def _entry(self, filename: str) -> dict[str, Any]:
        return self.entries.setdefault(filename, {})

    # -- cached dimensions -------------------------------------------------

    def size(self, filename: str, token: str) -> tuple[int, int] | None:
        """Cached ``(width, height)``, or ``None`` if it must be re-read.

        Keyed by the same token the thumbnails use, so a resize invalidates
        the cached dimensions along with the cached thumbnail rather than
        leaving the gallery reporting the old size.
        """
        entry = self.get(filename)
        if entry.get("size_token") != token:
            return None
        width, height = entry.get("width"), entry.get("height")
        if isinstance(width, int) and isinstance(height, int):
            return width, height
        return None

    def set_size(self, filename: str, token: str, width: int, height: int) -> None:
        entry = self._entry(filename)
        entry["width"] = width
        entry["height"] = height
        entry["size_token"] = token

    def __contains__(self, filename: object) -> bool:
        return filename in self.entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    # -- keeping up with operations ---------------------------------------

    def apply_rename(self, mapping: dict[str, str]) -> None:
        """Follow a rename. Called by the rename op, so ratings survive it."""
        renamed = {mapping.get(name, name): entry for name, entry in self.entries.items()}
        self.entries = renamed

    def apply_copies(self, mapping: dict[str, list[str]]) -> None:
        """Follow an augmentation: each new variant inherits its source's entry."""
        for source, variants in mapping.items():
            entry = self.entries.get(source)
            if not entry:
                continue
            for variant in variants:
                self.entries.setdefault(variant, dict(entry))

    def prune(self, keep: set[str]) -> int:
        """Drop entries for images that no longer exist. Returns the count."""
        stale = [name for name in self.entries if name not in keep]
        for name in stale:
            del self.entries[name]
        return len(stale)


def _read_v1_descriptions(root: Path) -> dict[str, dict[str, Any]]:
    """Salvage quality ratings from v1's ``*_descriptions.json``.

    Captions are deliberately *not* imported: the ``.txt`` sidecars are the
    truth, and importing a stale JSON caption would recreate exactly the
    two-sources-of-truth problem this file exists to end.
    """
    entries: dict[str, dict[str, Any]] = {}
    try:
        candidates = sorted(root.glob("*_descriptions.json"))
    except OSError:
        return entries

    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for record in data:
            if not isinstance(record, dict):
                continue
            name = record.get("fileName")
            quality = record.get("quality")
            if name and quality in QUALITY_VALUES:
                entries.setdefault(str(name), {})["quality"] = quality
    return entries
