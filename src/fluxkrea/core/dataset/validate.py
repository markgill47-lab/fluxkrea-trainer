"""Validation: report everything wrong with a dataset, change nothing.

"This is the thing that would have caught every v1 dataset bug before it
reached a training run, and it is cheap" (doc 03).

The mask checks matter most. ai-toolkit sets ``has_mask_image`` only if it
finds a basename match and otherwise trains the image unmasked *without
warning*; when a mask's dimensions disagree with its image it warns and
tries to swap the sizes (``dataloader_mixins.py:1473``). Both failures are
silent enough to survive a whole training run, so they get caught here.
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths
from ..events import Emitter, Log, no_op, safe
from ..imaging import ImageError, Size, has_grey, read_size
from ..imaging import validate_image as _validate_image
from .item import MASK_SUFFIX, DatasetItem
from .scan import natural_key, scan

ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass(frozen=True, slots=True)
class Problem:
    """One finding. ``kind`` is stable and machine-readable; the API sends it."""

    kind: str
    severity: str
    message: str
    stem: str | None = None
    path: Path | None = None

    def __str__(self) -> str:
        where = f"{self.stem}: " if self.stem else ""
        return f"[{self.severity}] {where}{self.message}"


@dataclass
class ValidationReport:
    root: Path
    items: int = 0
    problems: list[Problem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """No errors. Warnings are for the human, not for the gate."""
        return not self.errors

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == ERROR]

    @property
    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == WARNING]

    def of_kind(self, kind: str) -> list[Problem]:
        return [p for p in self.problems if p.kind == kind]

    def counts(self) -> dict[str, int]:
        totals: dict[str, int] = defaultdict(int)
        for problem in self.problems:
            totals[problem.kind] += 1
        return dict(totals)

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root.as_posix(),
            "items": self.items,
            "ok": self.ok,
            "counts": self.counts(),
            "problems": [
                {
                    "kind": p.kind,
                    "severity": p.severity,
                    "message": p.message,
                    "stem": p.stem,
                    "path": p.path.as_posix() if p.path else None,
                }
                for p in self.problems
            ],
        }

    def summary(self) -> str:
        if not self.problems:
            return f"{self.items} items, no problems"
        return (
            f"{self.items} items, {len(self.errors)} errors, "
            f"{len(self.warnings)} warnings"
        )


def validate(
    root: str | os.PathLike[str],
    *,
    items: Sequence[DatasetItem] | None = None,
    min_resolution: int = 512,
    require_masks: bool = False,
    extensions: Iterable[str] | None = None,
    caption_ext: str = ".txt",
    emit: Emitter = no_op,
    cancel: threading.Event | None = None,
) -> ValidationReport:
    """Inspect a dataset. Reports; never modifies.

    ``require_masks`` turns "this image has no mask" from silence into an
    error - on for a masked training run, off for an ordinary one.
    """
    emit = safe(emit)
    folder = paths.expand(root)
    found = list(items) if items is not None else scan(
        folder, extensions=extensions, caption_ext=caption_ext, cancel=cancel
    )
    report = ValidationReport(root=folder, items=len(found))
    add = report.problems.append

    known_stems: dict[str, list[DatasetItem]] = defaultdict(list)

    for item in found:
        known_stems[item.stem].append(item)

        ok, message = _validate_image(item.image, min_edge=None)
        if not ok:
            add(Problem("unreadable_image", ERROR, message, item.stem, item.image))
            continue

        try:
            size = read_size(item.image)
        except ImageError as exc:
            add(Problem("unreadable_image", ERROR, str(exc), item.stem, item.image))
            continue

        if size.shortest < min_resolution:
            add(
                Problem(
                    "below_resolution_floor",
                    WARNING,
                    f"{size} is below the {min_resolution}px floor",
                    item.stem,
                    item.image,
                )
            )

        _check_caption(item, add)
        _check_mask(item, size, require_masks, add)

    _check_orphans(folder, found, caption_ext, add)
    _check_stem_collisions(known_stems, add)

    emit(Log(line=report.summary(), level=INFO if report.ok else WARNING))
    return report


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------


def _check_caption(item: DatasetItem, add) -> None:
    if item.caption is None:
        add(Problem("missing_caption", WARNING, "no caption sidecar", item.stem, item.image))
        return
    if not item.read_caption():
        add(Problem("empty_caption", WARNING, "caption file is empty", item.stem, item.caption))


def _check_mask(item: DatasetItem, size: Size, require_masks: bool, add) -> None:
    if item.mask is None:
        if require_masks:
            add(
                Problem(
                    "missing_mask",
                    ERROR,
                    "no mask; ai-toolkit would train this image unmasked and say nothing",
                    item.stem,
                    item.image,
                )
            )
        return

    if item.mask.suffix.lower() != MASK_SUFFIX:
        add(
            Problem(
                "mask_wrong_format",
                ERROR,
                f"mask is {item.mask.suffix} - masks are PNG, JPEG ringing greys the edges",
                item.stem,
                item.mask,
            )
        )

    try:
        mask_size = read_size(item.mask)
    except ImageError as exc:
        add(Problem("unreadable_mask", ERROR, str(exc), item.stem, item.mask))
        return

    if mask_size.as_tuple() != size.as_tuple():
        add(
            Problem(
                "mask_size_mismatch",
                ERROR,
                f"mask is {mask_size} but the image is {size}",
                item.stem,
                item.mask,
            )
        )


def _check_orphans(
    folder: Path,
    found: Sequence[DatasetItem],
    caption_ext: str,
    add,
) -> None:
    """Sidecars with no image - the signature of a half-finished rename."""
    known_stems = {item.stem for item in found}
    ext = caption_ext if caption_ext.startswith(".") else f".{caption_ext}"

    try:
        for entry in sorted(folder.iterdir(), key=lambda p: natural_key(p.name)):
            if entry.is_file() and entry.suffix.lower() == ext and entry.stem not in known_stems:
                add(
                    Problem(
                        "orphan_caption",
                        WARNING,
                        "caption with no image",
                        entry.stem,
                        entry,
                    )
                )
    except OSError:
        pass

    masks_root = paths.masks_dir(folder)
    if not masks_root.is_dir():
        return
    try:
        entries = sorted(masks_root.iterdir(), key=lambda p: natural_key(p.name))
    except OSError:
        return

    for entry in entries:
        if not entry.is_file():
            continue
        if entry.stem not in known_stems:
            add(Problem("orphan_mask", WARNING, "mask with no image", entry.stem, entry))
            continue
        if has_grey_safe(entry):
            add(
                Problem(
                    "mask_has_grey",
                    INFO,
                    "mask contains partial weights - deliberate feathering, or a bad resize",
                    entry.stem,
                    entry,
                )
            )


def _check_stem_collisions(known_stems: dict[str, list[DatasetItem]], add) -> None:
    """Two images with one basename, or two that differ only in case.

    The second is the Linux trap: ``Punch_001.jpg`` and ``punch_001.jpg``
    coexist on Windows and collide the moment the folder is rsynced to a
    fleet node, taking one image's mask with them.
    """
    for stem, items in sorted(known_stems.items(), key=lambda kv: natural_key(kv[0])):
        if len(items) > 1:
            names = ", ".join(i.image.name for i in items)
            add(
                Problem(
                    "duplicate_stem",
                    ERROR,
                    f"{len(items)} images share the basename {stem!r} ({names}); "
                    "they cannot have distinct captions or masks",
                    stem,
                    items[0].image,
                )
            )

    folded: dict[str, list[str]] = defaultdict(list)
    for stem in known_stems:
        folded[stem.lower()].append(stem)
    for lowered, variants in sorted(folded.items()):
        if len(variants) > 1:
            add(
                Problem(
                    "case_collision",
                    WARNING,
                    f"{sorted(variants)} differ only by case; they collide on a "
                    "case-insensitive filesystem and will not survive a sync",
                    lowered,
                )
            )


def has_grey_safe(path: Path) -> bool:
    from PIL import Image

    try:
        with Image.open(path) as mask:
            mask.load()
            return has_grey(mask)
    except Exception:  # noqa: BLE001 - unreadable masks are reported elsewhere
        return False


__all__ = ["ERROR", "INFO", "WARNING", "Problem", "ValidationReport", "validate"]