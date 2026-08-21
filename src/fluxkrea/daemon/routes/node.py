"""Node and health endpoints, plus the filesystem browser.

``GET /fs/browse`` exists because the web client has no native file dialog
(doc 02, "UI layer"). Paths come from the API, scoped to the configured
roots - which is also what stops the endpoint being a file browser for the
whole machine.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query

from ...core import paths
from ..security import Denied, check_path, is_loopback
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
    """List folders and image counts, for a client with no OS picker.

    **Browsing is not the same permission as reading.** The roots scope
    what a dataset may be *registered* from - `check_path` still enforces
    that on every operation. Confining the browser to them as well left a
    local operator unable to reach another drive at all: the work was on
    D:, the picker showed C:, and `dataset.roots` is not writable over this
    API, so there was no way out from inside the app.

    So on a loopback-bound daemon the browser goes anywhere, and the top
    level lists the machine's drives alongside the configured roots. That
    concedes nothing: somebody on the loopback interface is already on the
    machine and can read it with a file manager. A daemon listening wider
    already requires a token and stays confined to its roots.
    """
    roots = state.config.dataset.roots
    local = is_loopback(state.config.daemon.host)

    if path is None:
        entries = [_folder_entry(r, state) for r in roots if r.is_dir()]
        # The drives, so "my work is on D:" has an answer. Listed after the
        # roots, which are the answer most of the time.
        listed = {r.resolve() for r in roots if r.is_dir()}
        if local:
            entries.extend(
                _folder_entry(drive, state)
                for drive in _drives()
                if drive.resolve() not in listed
            )
        if entries:
            return {
                "path": None,
                "parent": None,
                "roots": [r.as_posix() for r in roots],
                "entries": entries,
            }
        target = paths.home()
    else:
        target = paths.expand(path) if local else check_path(state.config, path)

    if not target.is_dir():
        raise Denied(f"{target.as_posix()} is not a folder", status=404)

    entries = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if child.is_dir() and not child.name.startswith("."):
                entries.append(_folder_entry(child, state))
    except OSError as exc:
        raise Denied(f"cannot read {target.as_posix()}: {exc}", status=403) from exc

    parent = target.parent
    # `parent == target` at a drive root, which is where "up" means "back to
    # the list of drives" - reported as None so the client offers that.
    allowed_parent = parent != target and (
        local or not roots or any(paths.is_within(parent, root) for root in roots)
    )

    return {
        "path": target.as_posix(),
        "parent": parent.as_posix() if allowed_parent else None,
        "roots": [r.as_posix() for r in roots],
        "entries": entries,
    }


def _drives() -> list[Path]:
    """Every drive with a filesystem on it, on Windows; ``/`` elsewhere."""
    if os.name != "nt":
        return [Path("/")]

    import string

    found = []
    for letter in string.ascii_uppercase:
        drive = Path(f"{letter}:/")
        try:
            if drive.is_dir():
                found.append(drive)
        except OSError:
            continue
    return found


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
        # A drive root has no `name` - `Path("D:/").name` is "" - and an
        # unnamed row in a picker is unclickable.
        "name": folder.name or folder.as_posix(),
        "images": images,
        "has_masks": (folder / paths.MASKS_DIRNAME).is_dir(),
        "dataset_id": registered.id if registered else None,
    }
