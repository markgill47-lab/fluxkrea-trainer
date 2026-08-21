"""Server-side thumbnails, cached.

Doc 10 is firm about this: the client never receives a 2K training image
to draw a 160px cell. Over an SSH tunnel that is the difference between a
usable grid and an unusable one - a 210-image dataset at 3MB each is
630MB to scroll past, against about 2MB of thumbnails.

Cached beside nothing: they live in the app's cache directory, not in the
dataset folder, because a dataset folder is training data and gets rsynced
to the fleet. Derived files do not belong in it.

Cache keys are content tokens - source size and mtime, hashed - so a
re-masked or re-resized image gets a different URL and the browser's cache
busts itself. There is no invalidation logic anywhere, which is the point.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from PIL import Image

from .. import paths
from ..imaging import ImageError, load_oriented

#: The two sizes doc 10 specifies: a grid cell and an inspector preview.
SIZES = (160, 480)

#: WebP at this quality holds up at both sizes and stays inside the
#: per-thumbnail budget (12KB at 160px, 40KB at 480px).
QUALITY = 75


def token_for(source: Path) -> str:
    """A short cache token for a file, from its size and mtime.

    Deliberately not a content hash: hashing 10,000 training images to
    build a grid would cost more than sending them. Size and mtime change
    whenever an operation rewrites a file, which is the only case that
    matters here.
    """
    try:
        stat = source.stat()
    except OSError:
        return "0"
    raw = f"{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def cache_path(dataset_id: str, stem: str, size: int, token: str) -> Path:
    """Where a thumbnail lives. One folder per dataset, flat within it."""
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in stem)[:120]
    return paths.cache_dir() / "thumbs" / dataset_id / f"{safe}-{size}-{token}.webp"


def build(
    source: str | os.PathLike[str],
    dataset_id: str,
    stem: str,
    size: int = 160,
) -> Path:
    """Return a cached thumbnail, generating it if it is not there yet.

    Raises :class:`ImageError` if the source cannot be read, so a broken
    file surfaces as a 404 on that one cell rather than a blank grid.
    """
    if size not in SIZES:
        raise ValueError(f"thumbnail size must be one of {SIZES}, got {size}")

    image_path = paths.expand(source)
    token = token_for(image_path)
    target = cache_path(dataset_id, stem, size, token)
    if target.is_file():
        return target

    image, _ = load_oriented(image_path)
    try:
        # EXIF orientation is already applied by load_oriented, so the
        # thumbnail is oriented the same way every other view of this image
        # is - a filmstrip cell that disagrees with the canvas is a bug
        # report waiting to happen.
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".webp.tmp")
        image.save(tmp, format="WEBP", quality=QUALITY, method=4)
        tmp.replace(target)
    finally:
        image.close()

    return target


def purge(dataset_id: str) -> int:
    """Drop a dataset's cached thumbnails. Returns how many were removed.

    Stale entries are otherwise harmless - a new token means a new
    filename, and the old one is simply never requested again - but a
    dataset that has been renamed a few times accumulates them.
    """
    folder = paths.cache_dir() / "thumbs" / dataset_id
    if not folder.is_dir():
        return 0

    removed = 0
    for entry in folder.glob("*.webp"):
        try:
            entry.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def ensure_many(
    items: list[tuple[Path, str]],
    dataset_id: str,
    size: int = 160,
) -> int:
    """Pre-generate thumbnails for a whole dataset. Returns how many were built.

    Called after a scan so the first scroll through a large dataset is not
    also the first generation pass. Failures are skipped, not raised: one
    corrupt image should not stop the other 209.
    """
    built = 0
    for source, stem in items:
        try:
            target = build(source, dataset_id, stem, size)
        except (ImageError, OSError, ValueError):
            continue
        if target.is_file():
            built += 1
    return built


__all__ = ["QUALITY", "SIZES", "build", "cache_path", "ensure_many", "purge", "token_for"]
