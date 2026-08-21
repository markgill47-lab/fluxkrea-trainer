"""Resize a dataset to a target longest edge. Was v1's ``fix_images``.

Carried forward from v1 (doc 01, ``d1890ce``):

* EXIF orientation baked into the pixels before any resize.
* File handles released, so a later rename does not fail on Windows.
* An image already at the target size and correctly oriented is *copied*,
  not re-encoded - a pointless lossy round trip otherwise.
* Optional upscaling of small images, rather than skipping them - and
  here it is the default, since a bucket of mixed-resolution images is
  worse for training than a few upscaled ones.

New here, and the reason this had to be rewritten rather than patched:
**masks are resized with the bundle, using NEAREST** (doc 03). A dataset
whose images were resized while its masks were not is a dataset where
every mask silently mismatches - and ai-toolkit's response to that is to
warn once and try to swap the dimensions.
"""

from __future__ import annotations

import os
import shutil
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ... import paths
from ...events import Emitter, Log, Progress, is_cancelled, no_op, safe
from ...imaging import (
    ImageError,
    Size,
    load_oriented,
    read_size,
    resize_longest,
    resize_mask,
    save_image,
    save_mask,
    target_size_for,
    validate_image,
)
from ..item import DatasetItem
from ..metadata import Metadata
from ..scan import scan


@dataclass
class ResizeResult:
    """What happened. ``items`` describes the dataset *after* the operation."""

    root: Path
    output: Path
    processed: int = 0
    skipped: int = 0
    masks_resized: int = 0
    #: Images left alone because ``upscale=False`` and they were smaller
    #: than the target. Not failures - the operation declined to invent
    #: pixels, on request.
    too_small: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    items: list[DatasetItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def total(self) -> int:
        return self.processed + self.skipped + len(self.too_small) + len(self.failed)

    def summary(self) -> str:
        parts = [f"{self.processed} resized", f"{self.skipped} already correct"]
        if self.masks_resized:
            parts.append(f"{self.masks_resized} masks")
        if self.too_small:
            parts.append(f"{len(self.too_small)} left alone (smaller than the target)")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts)

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root.as_posix(),
            "output": self.output.as_posix(),
            "processed": self.processed,
            "skipped": self.skipped,
            "masks_resized": self.masks_resized,
            "too_small": [{"stem": s, "reason": r} for s, r in self.too_small],
            "failed": [{"stem": s, "reason": r} for s, r in self.failed],
            "ok": self.ok,
        }


def resize(
    root: str | os.PathLike[str],
    target_longest: int,
    *,
    output: str | os.PathLike[str] | None = None,
    items: Sequence[DatasetItem] | None = None,
    upscale: bool = True,
    min_edge: int = 0,
    extensions: Iterable[str] | None = None,
    caption_ext: str = ".txt",
    emit: Emitter = no_op,
    cancel: threading.Event | None = None,
) -> ResizeResult:
    """Fit every image's longest edge to *target_longest*, aspect preserved.

    *output* of ``None`` means in place. *items* restricts the operation to
    a subset - the gallery's "resize selection" - and is the shape the API
    passes through.

    **Images smaller than the target are enlarged.** The target is the size
    the dataset is meant to be, and a bucket of mixed-resolution images is
    worse for training than a few upscaled ones. v1 refuses instead, by way
    of a hard 512 floor in its validator that rejects small images as
    though they were corrupt - a distinction still worth keeping, since a
    corrupt file *is* a failure. ``upscale=False`` restores the refusal for
    a caller that would rather see the list; those images are reported as
    ``too_small`` and left untouched, not counted as errors.

    Enlarging happens in a single LANCZOS pass to the final size.
    ``min_edge`` only warns: an image so wide that fitting its long edge
    still leaves the short one under the floor is worth knowing about, but
    it is not this operation's job to refuse it.
    """
    emit = safe(emit)
    source = paths.expand(root)
    destination = paths.expand(output) if output is not None else source
    in_place = destination == source

    if target_longest < 1:
        raise ValueError(f"target size must be positive, got {target_longest}")

    selected = list(items) if items is not None else scan(
        source, extensions=extensions, caption_ext=caption_ext, cancel=cancel
    )
    result = ResizeResult(root=source, output=destination)
    if not selected:
        emit(Log(line=f"No images found in {source}", level="warning"))
        return result

    if not in_place:
        paths.ensure_dir(destination)

    total = len(selected)
    emit(Progress(step=0, total=total, message="Resizing"))

    for index, item in enumerate(selected, start=1):
        if is_cancelled(cancel):
            emit(Log(line=f"Cancelled after {index - 1} of {total}", level="warning"))
            break

        target_item = item if in_place else item.rebased(destination)
        try:
            outcome = _resize_one(
                item,
                target_item,
                target_longest,
                upscale=upscale,
                min_edge=min_edge,
                in_place=in_place,
                emit=emit,
            )
        except TooSmall as exc:
            result.too_small.append((item.stem, str(exc)))
            emit(Log(line=f"{item.image.name}: {exc}", level="warning"))
            emit(Progress(step=index, total=total, message="Resizing"))
            continue
        except (ImageError, OSError) as exc:
            result.failed.append((item.stem, str(exc)))
            emit(Log(line=f"{item.image.name}: {exc}", level="error"))
            emit(Progress(step=index, total=total, message="Resizing"))
            continue

        resized_image, resized_mask, final = outcome
        if resized_image:
            result.processed += 1
        else:
            result.skipped += 1
        if resized_mask:
            result.masks_resized += 1
        result.items.append(final)

        emit(Progress(step=index, total=total, message="Resizing"))

    if not in_place:
        _carry_metadata(source, destination, result.items)

    emit(Log(line=result.summary(), level="info" if result.ok else "warning"))
    return result


class TooSmall(Exception):
    """The image would have to be enlarged to reach the target."""


def _resize_one(
    item: DatasetItem,
    target: DatasetItem,
    target_longest: int,
    *,
    upscale: bool,
    min_edge: int,
    in_place: bool,
    emit: Emitter,
) -> tuple[bool, bool, DatasetItem]:
    """Resize one bundle. Returns ``(image_written, mask_written, item)``."""
    ok, message = validate_image(item.image, min_edge=None)
    if not ok:
        raise ImageError(message)

    image, was_rotated = load_oriented(item.image)
    size = Size(*image.size)

    if size.longest < target_longest and not upscale:
        image.close()
        raise TooSmall(
            f"{size} is smaller than the {target_longest}px target; "
            "drop --no-upscale to enlarge it"
        )

    already_correct = size.longest == target_longest
    final_size = size if already_correct else target_size_for(size, target_longest)

    if min_edge and final_size.shortest < min_edge:
        # A frame wide enough that fitting its long edge still leaves the
        # short one under the floor. Worth saying; not worth refusing.
        emit(
            Log(
                line=(
                    f"{item.image.name}: {final_size} leaves the short edge under the "
                    f"{min_edge}px floor"
                ),
                level="warning",
            )
        )

    if already_correct and not was_rotated:
        # Nothing to do to the pixels. Copy rather than re-encode.
        image.close()
        if not in_place:
            _copy_bundle(item, target)
        return False, False, target

    # One resample, whichever direction it goes.
    image = image if already_correct else resize_longest(image, target_longest)
    save_image(image, target.image)
    image.close()

    _carry_caption(item, target, in_place)
    mask_written = _carry_mask(item, target, final_size, in_place)
    return True, mask_written, target


def _copy_bundle(item: DatasetItem, target: DatasetItem) -> None:
    """Copy every member of an untouched bundle to the output folder."""
    target.image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item.image, target.image)
    if item.caption and item.caption.is_file() and target.caption:
        shutil.copy2(item.caption, target.caption)
    if item.mask and item.mask.is_file() and target.mask:
        target.mask.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.mask, target.mask)


def _carry_caption(item: DatasetItem, target: DatasetItem, in_place: bool) -> None:
    if in_place or item.caption is None or target.caption is None:
        return
    if item.caption.is_file():
        target.caption.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.caption, target.caption)


def _carry_mask(item: DatasetItem, target: DatasetItem, size: Size, in_place: bool) -> bool:
    """Resize the mask to match its image exactly. NEAREST, never LANCZOS.

    A mask that is already the right size is copied rather than resampled,
    which is both faster and one fewer chance to introduce a grey pixel.
    """
    if item.mask is None or target.mask is None or not item.mask.is_file():
        return False

    from PIL import Image

    try:
        current = read_size(item.mask)
    except ImageError:
        current = None

    if current is not None and current.as_tuple() == size.as_tuple():
        if not in_place:
            target.mask.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.mask, target.mask)
        return False

    with Image.open(item.mask) as mask:
        mask.load()
        resized = resize_mask(mask, size)
    save_mask(resized, target.mask)
    return True


def _carry_metadata(source: Path, destination: Path, items: Sequence[DatasetItem]) -> None:
    """Copy the derived-metadata entries for whatever actually landed."""
    incoming = Metadata.load(source)
    if not len(incoming):
        return
    outgoing = Metadata.load(destination)
    for item in items:
        entry = incoming.get(item.image.name)
        if entry:
            outgoing.entries.setdefault(item.image.name, dict(entry))
    if len(outgoing):
        outgoing.save()
