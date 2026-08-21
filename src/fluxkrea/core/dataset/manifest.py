"""What a dataset folder contains, file by file, for sync and drift.

Two jobs (doc 06):

**Sync.** ``fk dataset push`` asks the target for its manifest, diffs it
against the local one, and transfers only what differs. The
``--sidecars-only`` case is the important one: images are large and
static, captions and masks are small and change constantly, so once the
images are on a node a re-masked pass moves kilobytes instead of
gigabytes.

**Drift.** Copies are independent and therefore drift. A digest per entry
lets ``fk fleet datasets`` show which nodes disagree. Detecting drift is
in scope; automatic reconciliation is not - the client reports and the
human decides which copy wins.

Paths are relative and POSIX, always. The same manifest is produced on a
Windows laptop and a Linux node, or the diff reports every file as changed
because one side spelled it with backslashes.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import paths
from ..config import DEFAULT_IMAGE_EXTENSIONS

#: Read size for hashing. Large enough that a 20MB image is a few reads.
CHUNK = 1 << 20

#: What ``--sidecars-only`` covers: the small, frequently-changing files.
SIDECAR_SUFFIXES = frozenset({".txt", ".json"})


@dataclass(frozen=True, slots=True)
class Entry:
    """One file. ``digest`` is empty when the manifest was built quick."""

    path: str
    size: int
    mtime: float
    digest: str = ""

    def same_content(self, other: Entry) -> bool:
        """Compare by digest where both have one, else by size and mtime.

        mtime comparison is deliberately coarse: FAT and some network
        filesystems keep two-second resolution, and rsync itself uses the
        same tolerance.
        """
        if self.digest and other.digest:
            return self.digest == other.digest
        return self.size == other.size and abs(self.mtime - other.mtime) <= 2.0

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"path": self.path, "size": self.size, "mtime": round(self.mtime, 3)}
        if self.digest:
            payload["digest"] = self.digest
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entry:
        return cls(
            path=str(data["path"]),
            size=int(data["size"]),
            mtime=float(data.get("mtime", 0.0)),
            digest=str(data.get("digest", "")),
        )


@dataclass
class Manifest:
    root: Path
    entries: dict[str, Entry] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[Entry]:
        return iter(sorted(self.entries.values(), key=lambda e: e.path))

    @property
    def bytes(self) -> int:
        return sum(entry.size for entry in self.entries.values())

    def diff(self, other: Manifest | Iterable[Entry]) -> Diff:
        """What would have to change on *other* to match this manifest."""
        target = other.entries if isinstance(other, Manifest) else {e.path: e for e in other}

        added = [e for path, e in self.entries.items() if path not in target]
        changed = [
            e
            for path, e in self.entries.items()
            if path in target and not e.same_content(target[path])
        ]
        removed = [e for path, e in target.items() if path not in self.entries]
        return Diff(
            added=sorted(added, key=lambda e: e.path),
            changed=sorted(changed, key=lambda e: e.path),
            removed=sorted(removed, key=lambda e: e.path),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root.as_posix(),
            "files": len(self.entries),
            "bytes": self.bytes,
            "entries": [entry.as_dict() for entry in self],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], root: Path | None = None) -> Manifest:
        entries = [Entry.from_dict(e) for e in data.get("entries", [])]
        return cls(
            root=root if root is not None else paths.expand(data.get("root", ".")),
            entries={e.path: e for e in entries},
        )


@dataclass(frozen=True, slots=True)
class Diff:
    added: list[Entry]
    changed: list[Entry]
    removed: list[Entry]

    @property
    def transfers(self) -> list[Entry]:
        """Files that have to move. Removals are reported, never acted on."""
        return sorted([*self.added, *self.changed], key=lambda e: e.path)

    @property
    def bytes(self) -> int:
        return sum(entry.size for entry in self.transfers)

    @property
    def in_sync(self) -> bool:
        return not self.added and not self.changed and not self.removed

    def summary(self) -> str:
        if self.in_sync:
            return "in sync"
        parts = []
        if self.added:
            parts.append(f"{len(self.added)} new")
        if self.changed:
            parts.append(f"{len(self.changed)} changed")
        if self.removed:
            parts.append(f"{len(self.removed)} only on the target")
        return ", ".join(parts) + f" ({human(self.bytes)} to send)"

    def as_dict(self) -> dict[str, Any]:
        return {
            "added": [e.as_dict() for e in self.added],
            "changed": [e.as_dict() for e in self.changed],
            "removed": [e.as_dict() for e in self.removed],
            "bytes": self.bytes,
            "in_sync": self.in_sync,
            "summary": self.summary(),
        }


def build(
    root: str | os.PathLike[str],
    *,
    digests: bool = True,
    sidecars_only: bool = False,
    extensions: Iterable[str] | None = None,
) -> Manifest:
    """Walk a dataset folder and describe every file that belongs to it.

    Only files the application owns are listed - images, captions, masks,
    previews and the two sidecar JSONs. Whatever else is in the folder is
    somebody's business, not the sync's.

    ``digests=False`` skips hashing, which turns a multi-gigabyte dataset
    from minutes into milliseconds at the cost of falling back to size and
    mtime comparison.
    """
    folder = paths.expand(root)
    if not folder.is_dir():
        raise NotADirectoryError(f"not a dataset folder: {folder}")

    allowed = {e.lower() for e in (extensions if extensions is not None else DEFAULT_IMAGE_EXTENSIONS)}
    manifest = Manifest(root=folder)

    for path in _walk(folder):
        relative = path.relative_to(folder).as_posix()
        suffix = path.suffix.lower()
        in_masks = relative.startswith(f"{paths.MASKS_DIRNAME}/")
        in_preview = relative.startswith(f"{paths.PREVIEW_DIRNAME}/")

        if sidecars_only:
            # Doc 06's cheap loop: captions and masks only.
            if not (in_masks or suffix in SIDECAR_SUFFIXES) or in_preview:
                continue
        elif not (in_masks or in_preview or suffix in allowed or suffix in SIDECAR_SUFFIXES):
            continue

        try:
            stat = path.stat()
        except OSError:
            continue

        manifest.entries[relative] = Entry(
            path=relative,
            size=stat.st_size,
            mtime=stat.st_mtime,
            digest=digest_of(path) if digests else "",
        )

    return manifest


def digest_of(path: Path) -> str:
    """sha256 of a file's contents. Streamed - datasets contain big files."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def _walk(folder: Path) -> Iterator[Path]:
    yield from _files_in(folder)
    for name in (paths.MASKS_DIRNAME, paths.PREVIEW_DIRNAME):
        sub = folder / name
        if sub.is_dir():
            yield from _files_in(sub)


def _files_in(folder: Path) -> Iterator[Path]:
    try:
        for entry in sorted(folder.iterdir(), key=lambda p: p.name):
            if entry.is_file() and not entry.name.startswith("."):
                yield entry
    except OSError:
        return


def human(size: int) -> str:
    """Bytes as something a person reads at a glance."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


__all__ = ["Diff", "Entry", "Manifest", "build", "digest_of", "human"]
