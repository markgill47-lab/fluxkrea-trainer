"""Face masking. The feature that started the rewrite (doc 04).

Training a pose LoRA on martial arts references bakes the reference
subjects' faces into it, and at generation time that LoRA fights the
character LoRA for control of the face. Painting the faces grey or
blurring them does not help - whatever is in those pixels *is a training
signal*, so a grey ellipse teaches "this concept comes with a grey blob"
and a blur leaks skin tone, hair colour and jaw silhouette. Loss masking
is strictly better: the region contributes zero gradient, so the model
learns nothing there at all.

The contract, which ai-toolkit already supports unmodified:

* **Faces are BLACK, everything else WHITE.** White is weight 1, trained.
* **Mask dimensions equal the source image's**, at native size.
* **Same basename**, in ``masks/``: ``punch_014.jpg`` -> ``masks/punch_014.png``.

The pipeline is ``detect -> review -> export``, not ``detect -> export``.
Detectors miss turned-away heads, extreme tilt and motion blur, and a
missed face is the failure that defeats the feature - so export refuses to
run on unreviewed or zero-box images unless explicitly forced.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ... import paths
from ...detect.base import ELLIPSE, MANUAL, RECT, SHAPES, Box, Detector, DetectorError, get_detector
from ...events import Emitter, Log, Progress, is_cancelled, no_op, safe
from ...imaging import ImageError, Size, load_oriented, save_image, save_mask
from ..boxes import BoxStore, ReviewProgress
from ..item import DatasetItem
from ..scan import scan

#: Mask values. Named because the polarity is the one thing that must not
#: be guessed at: white is trained, black is ignored.
TRAINED = 255
IGNORED = 0

#: How dark a preview's redacted region is drawn. Previews are a review aid
#: only - the trainer reads the masks, never these.
PREVIEW_DARKEN = 0.15


@dataclass
class DetectResult:
    root: Path
    scanned: int = 0
    with_faces: int = 0
    boxes: int = 0
    #: Images the detector found nothing in. Doc 04 wants these loud: they
    #: are where the misses hide.
    empty: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        parts = [f"{self.boxes} faces in {self.with_faces}/{self.scanned} images"]
        if self.empty:
            parts.append(f"{len(self.empty)} with no detections")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts)

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root.as_posix(),
            "scanned": self.scanned,
            "with_faces": self.with_faces,
            "boxes": self.boxes,
            "empty": self.empty,
            "failed": [{"stem": s, "reason": r} for s, r in self.failed],
            "ok": self.ok,
        }


@dataclass
class ExportResult:
    root: Path
    written: int = 0
    previews: int = 0
    #: Images with no boxes, which get an all-white mask - trained in full.
    unmasked: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    items: list[DatasetItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.refused

    def summary(self) -> str:
        if self.refused:
            return (
                f"Refused: {len(self.refused)} images are unreviewed or have no boxes. "
                "Review them, or pass force to accept them as-is."
            )
        parts = [f"{self.written} masks written"]
        if self.previews:
            parts.append(f"{self.previews} previews")
        if self.unmasked:
            parts.append(f"{len(self.unmasked)} fully trained (no boxes)")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts)

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root.as_posix(),
            "written": self.written,
            "previews": self.previews,
            "unmasked": self.unmasked,
            "refused": self.refused,
            "failed": [{"stem": s, "reason": r} for s, r in self.failed],
            "ok": self.ok,
        }


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


def detect_faces(
    root: str | os.PathLike[str],
    *,
    detector: Detector | str = "yunet",
    items: Sequence[DatasetItem] | None = None,
    confidence: float = 0.5,
    nms: float = 0.3,
    workers: int = 4,
    only_missing: bool = False,
    shape: str = ELLIPSE,
    extensions: Iterable[str] | None = None,
    store: BoxStore | None = None,
    emit: Emitter = no_op,
    cancel: threading.Event | None = None,
) -> DetectResult:
    """Run a detector over a dataset and persist the boxes.

    Threaded, because detection is the slow step and the images are
    independent. Manual boxes are preserved (see
    :meth:`BoxStore.record_detection`), so re-running with a lower
    confidence never discards review work.

    ``only_missing`` skips images that already have boxes - the cheap way
    to extend a pass over newly added images.

    *shape* is what the boxes are filled as, not what the detector returns:
    a detector reports a bounding box and has no opinion about the region.
    Ellipse is the default because a face is one, and the four corners a
    rectangle adds are background the run would otherwise be told to learn
    nothing from.
    """
    emit = safe(emit)
    folder = paths.expand(root)
    engine = get_detector(detector, confidence=confidence, nms=nms) if isinstance(detector, str) else detector
    fill = shape if shape in SHAPES else RECT

    selected = list(items) if items is not None else scan(folder, extensions=extensions, cancel=cancel)
    boxes = store if store is not None else BoxStore.load(folder)
    result = DetectResult(root=folder)

    if only_missing:
        selected = [i for i in selected if i.image.name not in boxes]

    if not selected:
        emit(Log(line="Nothing to detect", level="warning"))
        return result

    total = len(selected)
    result.scanned = total
    emit(Progress(step=0, total=total, message="Detecting faces"))
    lock = threading.Lock()
    done = {"n": 0}

    def work(item: DatasetItem) -> None:
        if is_cancelled(cancel):
            return
        try:
            found = [replace(box, shape=fill) for box in engine.detect(_as_bgr(item.image))]
        except (DetectorError, ImageError, OSError) as exc:
            with lock:
                result.failed.append((item.stem, str(exc)))
            emit(Log(line=f"{item.image.name}: {exc}", level="error"))
        else:
            with lock:
                boxes.record_detection(item.image.name, found)
                result.boxes += len(found)
                if found:
                    result.with_faces += 1
                else:
                    result.empty.append(item.image.name)
        finally:
            with lock:
                done["n"] += 1
                position = done["n"]
            emit(Progress(step=position, total=total, message="Detecting faces"))

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, selected))
    else:
        for item in selected:
            work(item)

    boxes.save()

    if result.empty:
        emit(
            Log(
                line=(
                    f"{len(result.empty)} images have no detections. These are where "
                    "misses hide - review them first."
                ),
                level="warning",
            )
        )
    emit(Log(line=result.summary(), level="info" if result.ok else "warning"))
    return result


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_mask(
    size: Size | tuple[int, int],
    boxes: Iterable[Box],
    *,
    expand: float = 1.6,
    expand_up: float = 1.35,
    feather: int = 6,
    invert: bool = False,
) -> Image.Image:
    """Build one mask. White everywhere, black over each expanded box.

    Feathering is applied *here*, deliberately, in a mask that is otherwise
    hard-edged - never acquired accidentally through a resize (doc 03).
    """
    width, height = size.as_tuple() if isinstance(size, Size) else (int(size[0]), int(size[1]))
    if width < 1 or height < 1:
        raise ValueError(f"cannot render a {width}x{height} mask")

    regions = list(boxes)
    array = np.full((height, width), TRAINED, dtype=np.uint8)
    ellipses: list[Box] = []
    for box in regions:
        grown = box.expanded(expand, expand_up)
        if grown.is_ellipse:
            # Deliberately *not* clamped. Clamping an ellipse clamps its
            # bounding box, which moves the centre and squashes the axes -
            # a face at the edge of the frame would be masked by a
            # different ellipse from the one drawn in review. The drawing
            # below clips at the canvas edge instead, which is the same
            # ellipse with the off-frame part missing.
            ellipses.append(grown)
            continue
        clipped = grown.clamped(width, height)
        if clipped.area <= 0:
            continue
        array[clipped.y : clipped.bottom, clipped.x : clipped.right] = IGNORED

    mask = Image.fromarray(array, mode="L")

    if ellipses:
        draw = ImageDraw.Draw(mask)
        for grown in ellipses:
            if grown.w < 1 or grown.h < 1:
                continue
            # Pillow's ellipse is inclusive of both bounds, so the -1 keeps
            # a w-pixel-wide box w pixels wide rather than w+1.
            draw.ellipse(
                (grown.x, grown.y, grown.right - 1, grown.bottom - 1),
                fill=IGNORED,
            )

    if feather > 0 and regions:
        # GaussianBlur is applied to the whole mask; with a hard-edged
        # source the only place it can act is the box boundary, which is
        # exactly the gradient wanted.
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather / 2))

    if invert:
        mask = Image.eval(mask, lambda value: TRAINED - value)
    return mask


def render_preview(image: Image.Image, mask: Image.Image) -> Image.Image:
    """The redacted image a human eyeballs to check coverage.

    A review aid, written to ``preview/``. The trainer never reads it - it
    consumes the mask.
    """
    darkened = Image.eval(image.convert("RGB"), lambda value: int(value * PREVIEW_DARKEN))
    return Image.composite(image.convert("RGB"), darkened, mask.convert("L"))


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


def export_masks(
    root: str | os.PathLike[str],
    *,
    items: Sequence[DatasetItem] | None = None,
    expand: float = 1.6,
    expand_up: float = 1.35,
    feather: int = 6,
    invert: bool = False,
    write_previews: bool = True,
    require_review: bool = True,
    force: bool = False,
    extensions: Iterable[str] | None = None,
    store: BoxStore | None = None,
    emit: Emitter = no_op,
    cancel: threading.Event | None = None,
) -> ExportResult:
    """Write ``masks/*.png`` from the stored boxes. Instant; no detection.

    Changing the expansion factor re-renders every mask from the same
    boxes, which is why review work is never lost to a parameter change.

    With *require_review* on - the default - export refuses, with a
    listing, if any image is unreviewed or has zero boxes. *force*
    overrides it, for datasets where some frames genuinely contain no face.
    """
    emit = safe(emit)
    folder = paths.expand(root)
    boxes = store if store is not None else BoxStore.load(folder)
    selected = list(items) if items is not None else scan(folder, extensions=extensions, cancel=cancel)
    result = ExportResult(root=folder)

    if not selected:
        emit(Log(line=f"No images found in {folder}", level="warning"))
        return result

    if require_review and not force:
        blocked = [
            item.image.name
            for item in selected
            if not boxes.is_reviewed(item.image.name) or boxes.get(item.image.name).empty
        ]
        if blocked:
            result.refused = blocked
            emit(Log(line=result.summary(), level="error"))
            for name in blocked[:20]:
                reason = "no boxes" if boxes.get(name).empty else "unreviewed"
                emit(Log(line=f"  {name}: {reason}", level="error"))
            if len(blocked) > 20:
                emit(Log(line=f"  ... and {len(blocked) - 20} more", level="error"))
            return result

    total = len(selected)
    emit(Progress(step=0, total=total, message="Exporting masks"))

    for index, item in enumerate(selected, start=1):
        if is_cancelled(cancel):
            emit(Log(line=f"Cancelled after {index - 1} of {total}", level="warning"))
            break

        found = boxes.boxes(item.image.name)
        try:
            written = _export_one(
                item,
                found,
                expand=expand,
                expand_up=expand_up,
                feather=feather,
                invert=invert,
                write_previews=write_previews,
            )
        except (ImageError, OSError, ValueError) as exc:
            result.failed.append((item.stem, str(exc)))
            emit(Log(line=f"{item.image.name}: {exc}", level="error"))
            emit(Progress(step=index, total=total, message="Exporting masks"))
            continue

        result.written += 1
        result.previews += int(written)
        if not found:
            result.unmasked.append(item.image.name)
        result.items.append(item.with_mask_path(item.expected_mask()))
        emit(Progress(step=index, total=total, message="Exporting masks"))

    if result.unmasked:
        emit(
            Log(
                line=(
                    f"{len(result.unmasked)} images have no boxes and were written as "
                    "fully-trained masks"
                ),
                level="warning",
            )
        )
    emit(Log(line=result.summary(), level="info" if result.ok else "warning"))
    return result


def _export_one(
    item: DatasetItem,
    boxes: Sequence[Box],
    *,
    expand: float,
    expand_up: float,
    feather: int,
    invert: bool,
    write_previews: bool,
) -> bool:
    """Write one mask, and its preview. Returns whether a preview was written."""
    image, _ = load_oriented(item.image)
    try:
        size = Size(*image.size)
        mask = render_mask(
            size,
            boxes,
            expand=expand,
            expand_up=expand_up,
            feather=feather,
            invert=invert,
        )
        save_mask(mask, item.expected_mask())

        if write_previews and boxes:
            save_image(render_preview(image, mask), item.expected_preview())
            return True
        return False
    finally:
        image.close()


# --------------------------------------------------------------------------
# review support
# --------------------------------------------------------------------------


def review_progress(
    root: str | os.PathLike[str],
    *,
    items: Sequence[DatasetItem] | None = None,
    store: BoxStore | None = None,
    extensions: Iterable[str] | None = None,
) -> ReviewProgress:
    """``184/210 reviewed, 6 with no detections``."""
    folder = paths.expand(root)
    boxes = store if store is not None else BoxStore.load(folder)
    selected = list(items) if items is not None else scan(folder, extensions=extensions)
    return boxes.progress(item.image.name for item in selected)


def review_order(
    items: Sequence[DatasetItem],
    store: BoxStore,
) -> list[DatasetItem]:
    """Zero-detection images first, then unreviewed, then the rest.

    Doc 04: the images with no detections are where the misses hide, so
    they are the ones a human should see first, not last.
    """

    def rank(item: DatasetItem) -> tuple[int, str]:
        entry = store.get(item.image.name)
        if entry.empty:
            return (0, item.image.name)
        if not entry.reviewed:
            return (1, item.image.name)
        return (2, item.image.name)

    return sorted(items, key=rank)


def set_boxes(
    root: str | os.PathLike[str],
    filename: str,
    boxes: Iterable[Box],
    *,
    reviewed: bool | None = True,
    store: BoxStore | None = None,
) -> BoxStore:
    """Replace one image's boxes. The remote review pass (``PUT .../boxes``)."""
    folder = paths.expand(root)
    target = store if store is not None else BoxStore.load(folder)
    target.set_boxes(filename, boxes, reviewed=reviewed)
    target.save()
    return target


def _as_bgr(path: Path) -> np.ndarray:
    """Load an image as the BGR uint8 array OpenCV wants.

    Via PIL rather than ``cv2.imread``, so EXIF orientation is applied.
    Detecting on unrotated pixels would put every box in the wrong place on
    a phone photo, and the mask would land on empty background.
    """
    image, _ = load_oriented(path)
    try:
        return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    finally:
        image.close()


__all__ = [
    "MANUAL",
    "DetectResult",
    "ExportResult",
    "detect_faces",
    "export_masks",
    "render_mask",
    "render_preview",
    "review_order",
    "review_progress",
    "set_boxes",
]
