"""Dataset endpoints - doc 06's table, in order.

The box endpoints are the ones that matter most here: they are what lets
the face-mask review happen from the laptop against a dataset sitting on a
node. Without them, review is tethered to whichever machine holds the
files, and the whole fleet workflow loses the step that makes masking
trustworthy.
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from ...core import paths
from ...core.dataset import archive, manifest, validate
from ...core.dataset.boxes import BoxStore
from ...core.dataset.ops import augment, detect_faces, export_masks, plan_rename, resize
from ...core.dataset.ops.rename import execute as execute_rename
from ...core.detect import MANUAL, Box, DetectorError, get_detector
from ...core.events import Emitter, Log
from ..security import Denied
from ..state import State
from .deps import get_state

router = APIRouter(prefix="/datasets", tags=["datasets"])

#: Operations reachable through ``POST /datasets/{id}/ops/{name}``.
OPS = ("resize", "rename", "augment", "mask", "detect", "validate")


# --------------------------------------------------------------------------
# registration and listing
# --------------------------------------------------------------------------


@router.get("")
def list_datasets(state: State = Depends(get_state)) -> dict[str, Any]:
    return {"datasets": [d.as_dict() for d in state.registry.list()]}


@router.post("", status_code=201)
def register(
    payload: dict[str, Any] = Body(...),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    path = payload.get("path")
    if not path:
        raise Denied("a path is required", status=400)
    return state.register(
        str(path), payload.get("name"), create=bool(payload.get("create", False))
    ).as_dict()


@router.delete("/{dataset_id}")
def forget(dataset_id: str, state: State = Depends(get_state)) -> dict[str, Any]:
    """Deregister. Never touches the folder - only this node's list of it."""
    if not state.registry.forget(dataset_id):
        raise Denied(f"no dataset registered as {dataset_id!r}", status=404)
    return {"forgotten": dataset_id}


@router.post("/{dataset_id}/scan")
def rescan(dataset_id: str, state: State = Depends(get_state)) -> dict[str, Any]:
    items = state.items(dataset_id)
    report = validate(
        state.dataset_path(dataset_id),
        items=items,
        min_resolution=state.config.dataset.min_resolution,
    )
    return {"items": len(items), "validation": report.as_dict()}


@router.get("/{dataset_id}/items")
def items(dataset_id: str, state: State = Depends(get_state)) -> dict[str, Any]:
    boxes = BoxStore.load(state.dataset_path(dataset_id))
    found = state.items(dataset_id)
    return {
        "items": [
            {
                "stem": item.stem,
                "filename": item.image.name,
                "caption": item.read_caption() if item.has_caption() else None,
                "has_caption": item.has_caption(),
                "has_mask": item.has_mask(),
                "quality": item.quality,
                "boxes": len(boxes.boxes(item.image.name)),
                "reviewed": boxes.is_reviewed(item.image.name),
            }
            for item in found
        ],
        "review": boxes.progress(i.image.name for i in found).as_dict(),
    }


@router.get("/{dataset_id}/validate")
def validate_dataset(
    dataset_id: str,
    require_masks: bool = Query(default=False),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    return validate(
        state.dataset_path(dataset_id),
        items=state.items(dataset_id),
        min_resolution=state.config.dataset.min_resolution,
        require_masks=require_masks,
    ).as_dict()


# --------------------------------------------------------------------------
# per-item bytes
# --------------------------------------------------------------------------


@router.get("/{dataset_id}/items/{stem}/image")
def item_image(dataset_id: str, stem: str, state: State = Depends(get_state)) -> FileResponse:
    return _file(state.item(dataset_id, stem).image)


@router.get("/{dataset_id}/items/{stem}/mask")
def item_mask(dataset_id: str, stem: str, state: State = Depends(get_state)) -> FileResponse:
    item = state.item(dataset_id, stem)
    if not item.has_mask():
        raise Denied(f"{stem} has no mask", status=404)
    return _file(item.mask)


@router.get("/{dataset_id}/items/{stem}/preview")
def item_preview(dataset_id: str, stem: str, state: State = Depends(get_state)) -> FileResponse:
    preview = state.item(dataset_id, stem).expected_preview()
    if not preview.is_file():
        raise Denied(f"{stem} has no preview; export masks to produce one", status=404)
    return _file(preview)


@router.get("/{dataset_id}/items/{stem}/caption")
def get_caption(dataset_id: str, stem: str, state: State = Depends(get_state)) -> dict[str, Any]:
    item = state.item(dataset_id, stem)
    return {"stem": stem, "caption": item.read_caption()}


@router.put("/{dataset_id}/items/{stem}/caption")
def put_caption(
    dataset_id: str,
    stem: str,
    payload: dict[str, Any] = Body(...),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    item = state.item(dataset_id, stem)
    written = item.write_caption(str(payload.get("caption", "")), state.config.dataset.caption_ext)
    return {"stem": stem, "caption": written.read_caption()}


# --------------------------------------------------------------------------
# the review pass
# --------------------------------------------------------------------------


@router.get("/{dataset_id}/items/{stem}/boxes")
def get_boxes(dataset_id: str, stem: str, state: State = Depends(get_state)) -> dict[str, Any]:
    item = state.item(dataset_id, stem)
    store = BoxStore.load(state.dataset_path(dataset_id))
    entry = store.get(item.image.name)
    return {"stem": stem, "filename": item.image.name, **entry.as_dict()}


@router.put("/{dataset_id}/items/{stem}/boxes")
def put_boxes(
    dataset_id: str,
    stem: str,
    payload: dict[str, Any] = Body(...),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """Replace an image's boxes. **The remote review pass** (doc 06)."""
    item = state.item(dataset_id, stem)
    try:
        boxes = [Box.from_dict(entry) for entry in payload.get("boxes", [])]
    except ValueError as exc:
        raise Denied(str(exc), status=400) from exc

    # A box arriving from a client without a source is one a human drew.
    boxes = [b if b.src not in ("", "unknown") else Box(b.x, b.y, b.w, b.h, MANUAL, b.conf) for b in boxes]

    store = BoxStore.load(state.dataset_path(dataset_id))
    store.set_boxes(item.image.name, boxes, reviewed=bool(payload.get("reviewed", True)))
    store.save()
    return {"stem": stem, **store.get(item.image.name).as_dict()}


@router.get("/{dataset_id}/review")
def review(dataset_id: str, state: State = Depends(get_state)) -> dict[str, Any]:
    found = state.items(dataset_id)
    store = BoxStore.load(state.dataset_path(dataset_id))
    return store.progress(item.image.name for item in found).as_dict()


# --------------------------------------------------------------------------
# operations
# --------------------------------------------------------------------------


@router.post("/{dataset_id}/ops/{operation}", status_code=202)
def run_operation(
    dataset_id: str,
    operation: str,
    payload: dict[str, Any] = Body(default={}),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """Start a dataset operation. Returns a task id; watch it over SSE.

    Every operation the CLI can run is here, because a feature reachable
    only by clicking - or only locally - makes the fleet second-class.
    """
    if operation not in OPS:
        raise Denied(f"unknown operation {operation!r}; expected one of {', '.join(OPS)}", status=404)

    root = state.dataset_path(dataset_id)
    options = dict(payload or {})
    work = _build(state, operation, root, options)

    task = state.tasks.submit(
        f"dataset.{operation}",
        work,
        dataset_id=dataset_id,
        detail={"operation": operation, "options": options},
    )
    return task.as_dict()


def _build(state: State, operation: str, root: Any, options: dict[str, Any]):  # noqa: ANN202
    """Turn a request payload into the callable the task runner will run."""
    config = state.config
    extensions = config.dataset.image_extensions
    caption_ext = config.dataset.caption_ext

    if operation == "resize":
        size = int(_opt(options, "size", 0))
        if size < 1:
            raise Denied("resize needs a positive size", status=400)

        def run_resize(emit: Emitter, cancel: threading.Event) -> Any:
            return resize(
                root,
                size,
                output=options.get("output"),
                upscale=bool(_opt(options, "upscale", True)),
                min_edge=config.dataset.min_resolution,
                extensions=extensions,
                caption_ext=caption_ext,
                emit=emit,
                cancel=cancel,
            )

        return run_resize

    if operation == "rename":
        prefix = str(options.get("prefix", "")).strip()
        if not prefix:
            raise Denied("rename needs a prefix", status=400)

        def run_rename(emit: Emitter, cancel: threading.Event) -> Any:
            plan = plan_rename(
                root,
                prefix,
                start_index=options.get("start"),
                digits=options.get("digits"),
                scramble=bool(options.get("scramble", False)),
                seed=options.get("seed"),
                extensions=extensions,
                caption_ext=caption_ext,
            )
            if options.get("dry_run"):
                # Planning touches no files, so a preview is the plan itself.
                for line in plan.describe():
                    emit(Log(line=line))
                emit(Log(line=plan.summary(), level="info" if plan.ok else "error"))
                return plan
            return execute_rename(plan, emit=emit, cancel=cancel)

        return run_rename

    if operation == "augment":
        transforms = list(options.get("transforms") or [])

        def run_augment(emit: Emitter, cancel: threading.Event) -> Any:
            return augment(
                root,
                transforms,
                output=options.get("output"),
                extensions=extensions,
                caption_ext=caption_ext,
                emit=emit,
                cancel=cancel,
            )

        return run_augment

    if operation == "detect":
        detector = _detector(state, options)

        def run_detect(emit: Emitter, cancel: threading.Event) -> Any:
            return detect_faces(
                root,
                detector=detector,
                workers=int(_opt(options, "workers", 4)),
                only_missing=bool(options.get("only_missing", False)),
                extensions=extensions,
                emit=emit,
                cancel=cancel,
            )

        return run_detect

    if operation == "mask":
        detector = None if options.get("detect") is False else _detector(state, options)

        def run_mask(emit: Emitter, cancel: threading.Event) -> Any:
            if detector is not None:
                found = detect_faces(
                    root,
                    detector=detector,
                    extensions=extensions,
                    emit=emit,
                    cancel=cancel,
                )
                if not found.ok:
                    return found
            return export_masks(
                root,
                expand=float(_opt(options, "expand", config.mask.expand)),
                expand_up=float(_opt(options, "expand_up", config.mask.expand_up)),
                feather=int(_opt(options, "feather", config.mask.feather)),
                invert=bool(_opt(options, "invert", config.mask.invert)),
                write_previews=bool(_opt(options, "previews", config.mask.write_previews)),
                require_review=config.mask.require_review,
                force=bool(options.get("force", False)),
                extensions=extensions,
                emit=emit,
                cancel=cancel,
            )

        return run_mask

    def run_validate(emit: Emitter, cancel: threading.Event) -> Any:
        return validate(
            root,
            min_resolution=int(_opt(options, "min_resolution", config.dataset.min_resolution)),
            require_masks=bool(options.get("require_masks", False)),
            extensions=extensions,
            caption_ext=caption_ext,
            emit=emit,
            cancel=cancel,
        )

    return run_validate


def _detector(state: State, options: dict[str, Any]):  # noqa: ANN202
    try:
        return get_detector(
            str(options.get("detector") or state.config.mask.detector),
            confidence=float(_opt(options, "confidence", state.config.mask.confidence)),
            nms=state.config.mask.nms,
        )
    except DetectorError as exc:
        raise Denied(str(exc), status=422) from exc


def _opt(options: dict[str, Any], key: str, default: Any) -> Any:
    """Read an option, treating an explicit null as absent.

    A CLI sends every flag it knows about, unset ones included, so ``None``
    here means "not specified" rather than "set to nothing".
    """
    value = options.get(key)
    return default if value is None else value


# --------------------------------------------------------------------------
# sync
# --------------------------------------------------------------------------


@router.get("/{dataset_id}/manifest")
def get_manifest(
    dataset_id: str,
    digests: bool = Query(default=True),
    sidecars_only: bool = Query(default=False),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """Per-item size, mtime and digest, for sync and drift detection."""
    return manifest.build(
        state.dataset_path(dataset_id),
        digests=digests,
        sidecars_only=sidecars_only,
        extensions=state.config.dataset.image_extensions,
    ).as_dict()


@router.get("/{dataset_id}/export")
def export(
    dataset_id: str,
    sidecars_only: bool = Query(default=False),
    state: State = Depends(get_state),
) -> StreamingResponse:
    """Stream the dataset out as a tar."""
    root = state.dataset_path(dataset_id)
    members = [
        entry.path
        for entry in manifest.build(
            root,
            digests=False,
            sidecars_only=sidecars_only,
            extensions=state.config.dataset.image_extensions,
        )
    ]
    return StreamingResponse(
        archive.stream(root, members),
        media_type="application/x-tar",
        headers={"content-disposition": f'attachment; filename="{dataset_id}.tar"'},
    )


@router.post("/{dataset_id}/import")
async def import_tar(
    dataset_id: str,
    request: Request,
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """Take a tar stream and extract it into place.

    The transport that works from a Windows laptop with nothing installed -
    OpenSSH ships with Windows, rsync does not (doc 06).
    """
    root = state.dataset_path(dataset_id)
    body = await request.body()

    import io

    try:
        result = archive.extract(root, io.BytesIO(body))
    except archive.ArchiveError as exc:
        raise Denied(str(exc), status=400) from exc
    return result.as_dict()


def _file(path: Any) -> FileResponse:
    resolved = paths.expand(path)
    if not resolved.is_file():
        raise Denied(f"{resolved.name} is not there", status=404)
    return FileResponse(resolved)
