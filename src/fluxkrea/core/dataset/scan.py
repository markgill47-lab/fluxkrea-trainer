"""The one scanner. Folder in, ``DatasetItem`` list out.

v1 keeps two copies of the supported-extension tuple in two classes, which
is how its gallery and its processor come to disagree about what exists in
a folder (doc 01). There is exactly one here, it lives in the config, and
nothing outside this package globs a dataset folder for itself.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path

from .. import paths
from ..config import DEFAULT_IMAGE_EXTENSIONS
from ..events import Emitter, Log, is_cancelled, no_op, safe
from .item import CAPTION_SUFFIX, MASK_SUFFIX, DatasetItem
from .metadata import Metadata

_NUMERIC_RUN = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple[object, ...]:
    """Sort ``punch_2`` before ``punch_10``, and case-insensitively.

    Plain lexical sort puts ``_10`` before ``_2``, which makes a rename
    preview and a gallery both read as shuffled.
    """
    parts = _NUMERIC_RUN.split(name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def scan(
    root: str | os.PathLike[str],
    *,
    extensions: Iterable[str] | None = None,
    recursive: bool = False,
    caption_ext: str = CAPTION_SUFFIX,
    metadata: Metadata | None = None,
    emit: Emitter = no_op,
    cancel: threading.Event | None = None,
) -> list[DatasetItem]:
    """Every training bundle in *root*, in natural filename order.

    ``masks/`` and ``preview/`` are ours and are never scanned as training
    data - a recursive scan that picked up its own mask output would double
    the dataset with black-and-white images.
    """
    emit = safe(emit)
    folder = paths.expand(root)
    if not folder.is_dir():
        raise NotADirectoryError(f"not a dataset folder: {folder}")

    allowed = _normalise_extensions(extensions)
    meta = metadata if metadata is not None else Metadata.load(folder)
    masks = _mask_index(folder)

    items: list[DatasetItem] = []
    for image in _iter_images(folder, allowed, recursive=recursive, cancel=cancel):
        if is_cancelled(cancel):
            emit(Log(line="Scan cancelled", level="warning"))
            break
        caption = image.with_suffix(caption_ext if caption_ext.startswith(".") else f".{caption_ext}")
        items.append(
            DatasetItem(
                image=image,
                caption=caption if caption.is_file() else None,
                mask=masks.get(image.stem),
                quality=meta.quality(image.name),
            )
        )

    items.sort(key=lambda item: natural_key(str(item.image)))
    return items


def find(root: str | os.PathLike[str], stem: str, **kwargs: object) -> DatasetItem | None:
    """One item by basename - what the API's per-item endpoints resolve with."""
    for item in scan(root, **kwargs):  # type: ignore[arg-type]
        if item.stem == stem:
            return item
    return None


def stems(items: Iterable[DatasetItem]) -> list[str]:
    return [item.stem for item in items]


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


def _normalise_extensions(extensions: Iterable[str] | None) -> frozenset[str]:
    source = extensions if extensions is not None else DEFAULT_IMAGE_EXTENSIONS
    cleaned = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in source}
    return frozenset(cleaned)


def _iter_images(
    folder: Path,
    allowed: frozenset[str],
    *,
    recursive: bool,
    cancel: threading.Event | None,
) -> Iterator[Path]:
    managed = {p.resolve() for p in paths.managed_dirs(folder)}

    def walk(current: Path) -> Iterator[Path]:
        try:
            entries = sorted(current.iterdir(), key=lambda p: natural_key(p.name))
        except OSError:
            return
        for entry in entries:
            if is_cancelled(cancel):
                return
            if entry.is_dir():
                if recursive and entry.resolve() not in managed and not entry.name.startswith("."):
                    yield from walk(entry)
                continue
            if entry.suffix.lower() in allowed:
                yield entry

    yield from walk(folder)


def _mask_index(folder: Path) -> dict[str, Path]:
    """Map basename -> mask file, mirroring ai-toolkit's own lookup.

    ai-toolkit matches masks by basename from the sibling folder
    (``dataloader_mixins.py:1440``), so this must too - including picking up
    a mask with the wrong extension, which ``validate`` then complains
    about. Silently not seeing it would leave the image training unmasked,
    which is the exact failure the feature exists to prevent.
    """
    masks_root = paths.masks_dir(folder)
    if not masks_root.is_dir():
        return {}

    index: dict[str, Path] = {}
    try:
        entries = sorted(masks_root.iterdir(), key=lambda p: natural_key(p.name))
    except OSError:
        return {}

    for entry in entries:
        if not entry.is_file():
            continue
        existing = index.get(entry.stem)
        # Prefer the correct format when a folder holds both.
        if existing is None or (existing.suffix.lower() != MASK_SUFFIX and entry.suffix.lower() == MASK_SUFFIX):
            index[entry.stem] = entry
    return index
