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
from ...core.dataset import archive, manifest, thumbs, validate
from ...core.dataset.metadata import QUALITY_VALUES, Metadata
from ...core.dataset.boxes import BoxStore
from ...core.captioners import CaptionerError, DEFAULT_PROMPT, from_config as build_captioner
from ...core.dataset.ops import (
    augment,
    caption,
    detect_faces,
    export_masks,
    plan_rename,
    resize,
)
from ...core.dataset.ops.rename import execute as execute_rename
from ...core.detect import MANUAL, Box, DetectorError, get_detector
from ...core.imaging import ImageError, read_size
from ...core.events import Emitter, Log
from ..security import Denied
from ..state import State
from .deps import get_state

router = APIRouter(prefix="/datasets", tags=["datasets"])

#: Operations reachable through ``POST /datasets/{id}/ops/{name}``.
OPS = ("resize", "rename", "augment", "caption", "mask", "detect", "validate")


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
    """Every bundle, with the status the gallery paints on each cell.

    Dimensions are cached in ``metadata.json`` rather than read on every
    request: the gallery wants them for all 10,000 cells, and a header read
    per image per page load adds up. The cache key is the same token the
    thumbnails use, so a resize invalidates both together.
    """
    root = state.dataset_path(dataset_id)
    boxes = BoxStore.load(root)
    meta = Metadata.load(root)
    found = state.items(dataset_id)
    dirty = False

    payload = []
    for item in found:
        name = item.image.name
        token = thumbs.token_for(item.image)

        size = meta.size(name, token)
        if size is None:
            try:
                measured = read_size(item.image)
                size = (measured.width, measured.height)
                meta.set_size(name, token, *size)
                dirty = True
            except ImageError:
                size = None

        payload.append(
            {
                "stem": item.stem,
                "filename": name,
                "caption": item.read_caption() if item.has_caption() else None,
                "has_caption": item.has_caption(),
                "has_mask": item.has_mask(),
                "quality": item.quality,
                "boxes": len(boxes.boxes(name)),
                "reviewed": boxes.is_reviewed(name),
                "width": size[0] if size else None,
                "height": size[1] if size else None,
                # Changes when the file does, so a thumbnail URL carrying
                # it can be cached immutably (doc 10).
                "token": token,
            }
        )

    if dirty:
        meta.save()

    return {
        "items": payload,
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


@router.get("/{dataset_id}/items/{stem}/thumb")
def item_thumb(
    dataset_id: str,
    stem: str,
    size: int = Query(default=160),
    v: str | None = Query(default=None, description="cache token from the items listing"),
    state: State = Depends(get_state),
) -> FileResponse:
    """A cached thumbnail. The client never gets a full image for a cell.

    When the caller passes the item's token as ``v`` the response is
    immutable: a changed file yields a different token and therefore a
    different URL, so the browser cache busts itself and nothing here has
    to track invalidation.
    """
    item = state.item(dataset_id, stem)
    if size not in thumbs.SIZES:
        raise Denied(f"thumbnail size must be one of {thumbs.SIZES}", status=400)

    try:
        target = thumbs.build(item.image, dataset_id, stem, size)
    except (ImageError, OSError) as exc:
        raise Denied(f"cannot render a thumbnail for {stem}: {exc}", status=404) from exc

    headers = (
        {"cache-control": "public, max-age=31536000, immutable"}
        if v
        else {"cache-control": "no-cache"}
    )
    return FileResponse(target, media_type="image/webp", headers=headers)


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


@router.put("/{dataset_id}/items/{stem}/quality")
def put_quality(
    dataset_id: str,
    stem: str,
    payload: dict[str, Any] = Body(...),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """Set or clear an item's quality rating (good / ok / bad).

    Derived metadata, never authoritative for anything a trainer reads -
    delete ``metadata.json`` and nothing training-relevant is lost (doc 03).
    """
    item = state.item(dataset_id, stem)
    quality = payload.get("quality")
    if quality is not None and quality not in QUALITY_VALUES:
        raise Denied(f"quality must be one of {QUALITY_VALUES} or null", status=400)

    meta = Metadata.load(state.dataset_path(dataset_id))
    meta.set_quality(item.image.name, quality)
    meta.save()
    return {"stem": stem, "quality": quality}


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

    if operation == "caption":
        # Built here rather than inside the task so a bad provider name or
        # a missing SDK is a 422 on the request, not a task that starts
        # and immediately dies.
        captioner = _captioner(state, options)
        settings = config.captioner

        # A name wins over free text, and an unknown name falls back to the
        # default rather than to nothing - two hundred images described by
        # an empty prompt is a worse outcome than a typo deserves.
        prompt_text = state.prompts.resolve(
            options.get("prompt_name"),
            str(_opt(options, "prompt", settings.prompt) or DEFAULT_PROMPT),
        )

        def run_caption(emit: Emitter, cancel: threading.Event) -> Any:
            return caption(
                root,
                captioner,
                prompt=prompt_text,
                prefix=str(_opt(options, "prefix", settings.prefix)),
                overwrite=bool(_opt(options, "overwrite", False)),
                max_tokens=int(_opt(options, "max_tokens", settings.max_tokens)),
                extensions=extensions,
                caption_ext=caption_ext,
                emit=emit,
                cancel=cancel,
            )

        return run_caption

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


def _captioner(state: State, options: dict[str, Any]):  # noqa: ANN202
    """Build the captioner a caption request asks for.

    Overrides are accepted per request - the settings screen tests a
    provider before saving it - but the API key never travels this way.
    It comes from the environment or the keyring on the node, which is
    what keeps it out of request logs and out of ``config.toml``.
    """
    overrides = {
        key: options[key]
        for key in ("provider", "url", "model", "timeout")
        if options.get(key) is not None
    }
    try:
        return build_captioner(state.config.captioner, **overrides)
    except CaptionerError as exc:
        raise Denied(str(exc), status=422) from exc
    except TypeError as exc:
        # e.g. a `url` override aimed at the Claude backend, which has none.
        raise Denied(f"option not valid for this captioner: {exc}", status=422) from exc


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
