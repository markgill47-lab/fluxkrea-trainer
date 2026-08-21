"""Augmentation: flipped and rotated copies. Was v1's ``create_duplicates``.

The one substantive change from v1, and it is the whole reason masking
forced a rewrite: **a mask is transformed identically to its image**. A
flipped image needs a flipped mask, or the excluded region lands on the
wrong side of the frame and the face trains anyway - the exact failure
loss masking exists to prevent (doc 03).

Suffixes are kept byte-for-byte compatible with v1, including ``_flipVert``
for a 180-degree rotation. It is a misnomer, but existing datasets on the
fleet are already named that way and renaming them would break nothing
usefully.
"""

from __future__ import annotations

import os
import shutil
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ... import paths
from ...events import Emitter, Log, Progress, is_cancelled, no_op, safe
from ...imaging import ImageError, load_oriented, save_image, save_mask
from ..item import DatasetItem
from ..metadata import Metadata
from ..scan import scan


@dataclass(frozen=True, slots=True)
class Transform:
    """One augmentation. ``operation`` is ``None`` for a plain duplicate."""

    key: str
    label: str
    suffix: str
    operation: int | None

    def apply(self, image: Image.Image) -> Image.Image:
        """Applied identically to an image and to its mask."""
        if self.operation is None:
            return image.copy()
        return image.transpose(self.operation)


#: Keyed by the names v1's transformation dict used, so a saved preset from
#: v1 still means the same thing.
TRANSFORMS: dict[str, Transform] = {
    "flip_horizontal": Transform(
        "flip_horizontal", "Horizontal flip", "_flipHor", Image.Transpose.FLIP_LEFT_RIGHT
    ),
    "rotate_90_left": Transform("rotate_90_left", "90 left", "_rotLeft", Image.Transpose.ROTATE_90),
    "rotate_90_right": Transform(
        "rotate_90_right", "90 right", "_rotRight", Image.Transpose.ROTATE_270
    ),
    "rotate_180": Transform("rotate_180", "180", "_flipVert", Image.Transpose.ROTATE_180),
    "duplicate": Transform("duplicate", "Plain duplicate", "_dup", None),
}


@dataclass
class AugmentResult:
    root: Path
    output: Path
    created: int = 0
    masks: int = 0
    captions: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    items: list[DatasetItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        parts = [f"{self.created} images created"]
        if self.masks:
            parts.append(f"{self.masks} masks transformed")
        if self.captions:
            parts.append(f"{self.captions} captions copied")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts)

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root.as_posix(),
            "output": self.output.as_posix(),
            "created": self.created,
            "masks": self.masks,
            "captions": self.captions,
            "failed": [{"stem": s, "reason": r} for s, r in self.failed],
            "ok": self.ok,
        }


def augment(
    root: str | os.PathLike[str],
    transforms: Sequence[str | Transform],
    *,
    output: str | os.PathLike[str] | None = None,
    items: Sequence[DatasetItem] | None = None,
    extensions: Iterable[str] | None = None,
    caption_ext: str = ".txt",
    emit: Emitter = no_op,
    cancel: threading.Event | None = None,
) -> AugmentResult:
    """Write a transformed copy of every bundle, for each transform.

    With *output* set to a different folder the originals are copied across
    as well, so the result is a complete dataset rather than variants alone.
    An empty *transforms* means a plain duplicate, matching v1.
    """
    emit = safe(emit)
    source = paths.expand(root)
    destination = paths.expand(output) if output is not None else source
    in_place = destination == source

    chosen = _resolve(transforms)
    selected = list(items) if items is not None else scan(
        source, extensions=extensions, caption_ext=caption_ext, cancel=cancel
    )

    result = AugmentResult(root=source, output=destination)
    if not selected:
        emit(Log(line=f"No images found in {source}", level="warning"))
        return result

    paths.ensure_dir(destination)
    total = len(selected)
    emit(Progress(step=0, total=total, message="Augmenting"))
    copies: dict[str, list[str]] = {}

    for index, item in enumerate(selected, start=1):
        if is_cancelled(cancel):
            emit(Log(line=f"Cancelled after {index - 1} of {total}", level="warning"))
            break

        try:
            produced = _augment_one(item, chosen, destination, in_place, result)
        except (ImageError, OSError) as exc:
            result.failed.append((item.stem, str(exc)))
            emit(Log(line=f"{item.image.name}: {exc}", level="error"))
            emit(Progress(step=index, total=total, message="Augmenting"))
            continue

        copies[item.image.name] = [variant.image.name for variant in produced]
        result.items.extend(produced)
        emit(Progress(step=index, total=total, message="Augmenting"))

    _carry_metadata(source, destination, copies, in_place)
    emit(Log(line=result.summary(), level="info" if result.ok else "warning"))
    return result


def _augment_one(
    item: DatasetItem,
    transforms: Sequence[Transform],
    destination: Path,
    in_place: bool,
    result: AugmentResult,
) -> list[DatasetItem]:
    """Produce every variant of one bundle. Returns the new items only."""
    image, was_rotated = load_oriented(item.image)
    mask = _open_mask(item)
    produced: list[DatasetItem] = []

    try:
        if not in_place:
            original = item.rebased(destination)
            if was_rotated:
                # Orientation is baked in on the way out, exactly as resize
                # does it - the copy must not carry a tag its pixels ignore.
                save_image(image, original.image)
            else:
                original.image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item.image, original.image)
            result.created += 1
            _copy_caption(item, original, result)
            if mask is not None and original.mask is not None:
                save_mask(mask, original.mask)
                result.masks += 1
            produced.append(original)

        for transform in transforms:
            variant = item.suffixed(transform.suffix).rebased(destination)
            save_image(transform.apply(image), variant.image)
            result.created += 1
            _copy_caption(item, variant, result)

            if mask is not None and variant.mask is not None:
                # The same transform, on the mask. Not a resample: transpose
                # moves pixels without inventing values, so a hard edge stays
                # hard and no grey appears at the boundary.
                save_mask(transform.apply(mask), variant.mask)
                result.masks += 1

            produced.append(variant)
    finally:
        image.close()
        if mask is not None:
            mask.close()

    return produced


def _open_mask(item: DatasetItem) -> Image.Image | None:
    if item.mask is None or not item.mask.is_file():
        return None
    try:
        with Image.open(item.mask) as mask:
            mask.load()
            return mask.convert("L")
    except (OSError, ValueError) as exc:
        raise ImageError(f"cannot read mask {item.mask.name}: {exc}") from exc


def _copy_caption(item: DatasetItem, variant: DatasetItem, result: AugmentResult) -> None:
    """Every variant inherits the caption verbatim. It describes the content."""
    if item.caption is None or not item.caption.is_file():
        return
    target = variant.caption or variant.expected_caption(item.caption.suffix)
    if target.resolve() == item.caption.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item.caption, target)
    result.captions += 1


def _carry_metadata(
    source: Path, destination: Path, copies: dict[str, list[str]], in_place: bool
) -> None:
    """Variants inherit their source's quality rating."""
    incoming = Metadata.load(source)
    if not len(incoming):
        return
    target = incoming if in_place else Metadata.load(destination)
    if not in_place:
        for name in copies:
            entry = incoming.get(name)
            if entry:
                target.entries.setdefault(name, dict(entry))
    target.apply_copies(copies)
    if len(target):
        target.save()


def _resolve(transforms: Sequence[str | Transform]) -> list[Transform]:
    if not transforms:
        return [TRANSFORMS["duplicate"]]
    resolved: list[Transform] = []
    for entry in transforms:
        if isinstance(entry, Transform):
            resolved.append(entry)
            continue
        if entry not in TRANSFORMS:
            known = ", ".join(sorted(TRANSFORMS))
            raise ValueError(f"unknown transform {entry!r}; expected one of {known}")
        resolved.append(TRANSFORMS[entry])
    return resolved
