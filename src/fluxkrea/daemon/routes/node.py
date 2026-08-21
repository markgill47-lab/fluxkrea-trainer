"""Node and health endpoints, plus the filesystem browser.

``GET /fs/browse`` exists because the web client has no native file dialog
(doc 02, "UI layer"). Paths come from the API, scoped to the configured
roots - which is also what stops the endpoint being a file browser for the
whole machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query

from ...core import paths
from ..security import Denied, check_path
from ..state import State
from .deps import get_state

router = APIRouter(tags=["node"])


@router.get("/health")
def health(state: State = Depends(get_state)) -> dict[str, Any]:
    return state.health()


@router.get("/node")
def node(state: State = Depends(get_state)) -> dict[str, Any]:
    return state.node()


@router.get("/gpus")
def gpus(state: State = Depends(get_state)) -> dict[str, Any]:
    info = state.node()
    return {
        "gpus": info.get("gpus", []),
        "torch": info.get("torch"),
        "cuda": info.get("cuda"),
        "driver": info.get("driver"),
    }


@router.get("/models")
def models(state: State = Depends(get_state)) -> dict[str, Any]:
    """What this node can train, and whether the backend could run it.

    Worth asking before submitting: a node with no ai-toolkit checkout
    still knows the model list, and saying so up front beats a job that
    queues and then fails.
    """
    from ...core.backends.models import listing

    return {"models": listing(), "backends": state.backends()}


@router.get("/fs/browse")
def browse(
    path: str | None = Query(default=None, description="folder to list; defaults to the roots"),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """List folders and image counts, for a client with no OS picker."""
    roots = state.config.dataset.roots

    if path is None:
        if roots:
            return {
                "path": None,
                "parent": None,
                "roots": [r.as_posix() for r in roots],
                "entries": [_folder_entry(r, state) for r in roots if r.is_dir()],
            }
        # No roots configured means loopback-only and unrestricted; offer the
        # home directory as a starting point rather than nothing at all.
        target = paths.home()
    else:
        target = check_path(state.config, path)

    if not target.is_dir():
        raise Denied(f"{target.as_posix()} is not a folder", status=404)

    entries: list[dict[str, Any]] = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if child.is_dir() and not child.name.startswith("."):
                entries.append(_folder_entry(child, state))
    except OSError as exc:
        raise Denied(f"cannot read {target.as_posix()}: {exc}", status=403) from exc

    parent = target.parent
    allowed_parent = parent != target and (
        not roots or any(paths.is_within(parent, root) for root in roots)
    )

    return {
        "path": target.as_posix(),
        "parent": parent.as_posix() if allowed_parent else None,
        "roots": [r.as_posix() for r in roots],
        "entries": entries,
    }


def _folder_entry(folder: Path, state: State) -> dict[str, Any]:
    """A folder, with enough detail to tell datasets from ordinary folders."""
    extensions = {e.lower() for e in state.config.dataset.image_extensions}
    images = 0
    try:
        for child in folder.iterdir():
            if child.is_file() and child.suffix.lower() in extensions:
                images += 1
    except OSError:
        images = 0

    registered = state.registry.by_path(folder)
    return {
        "path": folder.as_posix(),
        "name": folder.name,
        "images": images,
        "has_masks": (folder / paths.MASKS_DIRNAME).is_dir(),
        "dataset_id": registered.id if registered else None,
    }
