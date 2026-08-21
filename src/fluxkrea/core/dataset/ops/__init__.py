"""Dataset operations. Every one of them goes through ``DatasetItem``.

Three conventions hold across all of them:

**Sidecars are never handled inline.** An operation asks the item what its
bundle contains. That is the whole point of doc 03 - v1 has three
hand-rolled copies of "and also handle the sidecar file", and the mask
made it three more chances to forget.

**Planning is separate from execution** wherever the operation is
destructive. A rename plan can be printed, diffed and refused before a
single file moves.

**Operations emit ``Progress`` and ``Log``, never ``Finished``.** The
runner that owns the operation - the CLI command, the daemon's task
worker - emits exactly one terminal event, so "one ``Finished`` per
operation" stays true no matter how operations are composed.
"""

from .augment import TRANSFORMS, AugmentResult, Transform, augment
from .caption import CaptionResult, caption
from .mask import (
    DetectResult,
    ExportResult,
    detect_faces,
    export_masks,
    render_mask,
    render_preview,
    review_order,
    review_progress,
    set_boxes,
)
from .rename import RenamePlan, RenameResult, plan_rename, rename
from .resize import ResizeResult, resize

__all__ = [
    "AugmentResult",
    "CaptionResult",
    "DetectResult",
    "ExportResult",
    "RenamePlan",
    "RenameResult",
    "ResizeResult",
    "TRANSFORMS",
    "Transform",
    "augment",
    "caption",
    "detect_faces",
    "export_masks",
    "plan_rename",
    "render_mask",
    "render_preview",
    "review_order",
    "review_progress",
    "set_boxes",
    "rename",
    "resize",
]
