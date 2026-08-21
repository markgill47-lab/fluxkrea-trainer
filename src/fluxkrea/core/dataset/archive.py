"""Tar streaming, for the transport that works with nothing installed.

Bulk data normally moves by rsync over SSH - incremental, compressed,
resumable. But OpenSSH ships with Windows 10+ and rsync does not, so the
laptop needs a path that works unaided: a tar stream into
``POST /datasets/{id}/import`` (doc 06, "moving bytes").

Extraction is the dangerous half. A tar is a list of paths chosen by
whoever built it, so ``extract`` here refuses anything that is not a plain
file landing inside the destination: no absolute paths, no ``..``, no
symlinks, no devices. Python's own ``data`` filter does most of this as of
3.12, and this adds the dataset-shaped checks on top.
"""

from __future__ import annotations

import io
import os
import tarfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from .. import paths

#: Read/write size for streaming. Matches the manifest's chunk.
CHUNK = 1 << 20


class ArchiveError(Exception):
    """A tar that will not be trusted, or cannot be built."""


@dataclass(frozen=True, slots=True)
class ImportResult:
    root: Path
    files: int
    bytes: int
    skipped: list[str]

    @property
    def ok(self) -> bool:
        return not self.skipped

    def summary(self) -> str:
        text = f"{self.files} files ({self.bytes} bytes) extracted"
        if self.skipped:
            text += f", {len(self.skipped)} refused"
        return text

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root.as_posix(),
            "files": self.files,
            "bytes": self.bytes,
            "skipped": self.skipped,
            "ok": self.ok,
        }


def stream(
    root: str | os.PathLike[str],
    members: Iterable[str] | None = None,
    *,
    chunk: int = CHUNK,
) -> Iterator[bytes]:
    """Yield an uncompressed tar of *members*, relative POSIX paths.

    Uncompressed on purpose: the payload is overwhelmingly JPEG and PNG,
    which do not compress, and gzip would burn CPU on both ends for
    nothing. ``members`` of ``None`` sends everything the manifest covers.
    """
    folder = paths.expand(root)
    if not folder.is_dir():
        raise NotADirectoryError(f"not a dataset folder: {folder}")

    if members is None:
        from .manifest import build

        members = [entry.path for entry in build(folder, digests=False)]

    buffer = _StreamBuffer()
    with tarfile.open(fileobj=buffer, mode="w|", format=tarfile.PAX_FORMAT) as archive:
        for relative in members:
            source = _safe_join(folder, relative)
            if source is None or not source.is_file():
                continue
            info = archive.gettarinfo(str(source), arcname=relative)
            # Ownership is meaningless across machines and often unmappable
            # on the far side; strip it rather than have extraction complain.
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with source.open("rb") as handle:
                archive.addfile(info, handle)
            yield from buffer.drain(chunk)
    yield from buffer.drain(0)


def extract(root: str | os.PathLike[str], source: BinaryIO) -> ImportResult:
    """Extract a tar stream into *root*, refusing anything unsafe.

    Refused members are reported rather than silently dropped - a client
    that sent something odd should be told, and a partially-applied import
    that says nothing is worse than one that lists what it would not take.
    """
    folder = paths.ensure_dir(paths.expand(root))
    resolved_root = folder.resolve()
    files = 0
    written = 0
    skipped: list[str] = []

    try:
        with tarfile.open(fileobj=source, mode="r|*") as archive:
            for member in archive:
                if not member.isfile():
                    # Directories are implied by the files inside them;
                    # symlinks and devices have no business here at all.
                    if not member.isdir():
                        skipped.append(f"{member.name} (not a regular file)")
                    continue

                target = _safe_join(folder, member.name)
                if target is None or not _inside(target, resolved_root):
                    skipped.append(f"{member.name} (outside the dataset folder)")
                    continue

                extracted = archive.extractfile(member)
                if extracted is None:
                    skipped.append(f"{member.name} (unreadable)")
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as handle:
                    while chunk := extracted.read(CHUNK):
                        handle.write(chunk)
                        written += len(chunk)
                files += 1
    except tarfile.TarError as exc:
        raise ArchiveError(f"not a readable tar stream: {exc}") from exc

    return ImportResult(root=folder, files=files, bytes=written, skipped=skipped)


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


def _safe_join(root: Path, relative: str) -> Path | None:
    """Join a tar member name onto the root, or ``None`` if it escapes."""
    name = relative.replace("\\", "/").strip()
    if not name or name.startswith("/"):
        return None
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in ("..", "") for part in pure.parts):
        return None
    if len(pure.parts) > 1 and pure.parts[0] not in (paths.MASKS_DIRNAME, paths.PREVIEW_DIRNAME):
        # A dataset is flat plus two known subfolders. Anything deeper is
        # either a mistake or an attempt to write somewhere interesting.
        return None
    return root.joinpath(*pure.parts)


def _inside(target: Path, resolved_root: Path) -> bool:
    try:
        # The parent is resolved rather than the target, which may not exist
        # yet; on Windows resolving a missing path can invent a prefix.
        parent = target.parent
        parent.mkdir(parents=True, exist_ok=True)
        return parent.resolve().is_relative_to(resolved_root)
    except OSError:
        return False


class _StreamBuffer(io.RawIOBase):
    """A file-like sink that hands written bytes back out in chunks.

    ``tarfile`` in stream mode writes to a file object; the HTTP layer wants
    an iterator of bytes. This is the adapter, and it keeps memory flat -
    only the current chunk is held, never the whole archive.
    """

    def __init__(self) -> None:
        self._parts: list[bytes] = []
        self._size = 0

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:  # type: ignore[override]
        payload = bytes(data)
        self._parts.append(payload)
        self._size += len(payload)
        return len(payload)

    def drain(self, threshold: int) -> Iterator[bytes]:
        if self._size and self._size >= threshold:
            payload = b"".join(self._parts)
            self._parts.clear()
            self._size = 0
            yield payload


__all__ = ["ArchiveError", "ImportResult", "extract", "stream"]
