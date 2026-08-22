"""Project endpoints - the grouping a room of students works in.

Deliberately no "current project" on the node. A project is chosen by a
browser and lives in that browser, because the whole point of the lab
deployment is several people on one daemon at once: a server-side
selection would mean the last student to click changed everybody's screen.

So the daemon stores projects and the client remembers which one it has
open. Every endpoint here is addressed by id, and none of them has an
implicit subject.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from ..projects import ProjectError
from ..security import Denied
from ..state import State
from .deps import get_state

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("")
def list_projects(state: State = Depends(get_state)) -> dict[str, Any]:
    """Every project, with its datasets resolved against the registry.

    The resolved list is what the client renders, and it is not the same
    as the stored one: a dataset deregistered from under a project is
    dropped here rather than shown as a row that 404s on click. The stored
    ids stay put, so re-registering the folder brings it back.
    """
    return {
        "projects": [_payload(state, project.id) for project in state.projects.list()],
        "count": len(state.projects),
    }


@router.post("", status_code=201)
def create_project(
    payload: dict[str, Any] = Body(default={}),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    datasets = [str(d) for d in (payload.get("datasets") or [])]
    try:
        project = state.projects.create(str(payload.get("name") or ""), datasets)
    except ProjectError as exc:
        raise Denied(str(exc), status=422) from exc
    return _payload(state, project.id)


@router.get("/{project_id}")
def get_project(project_id: str, state: State = Depends(get_state)) -> dict[str, Any]:
    state.project(project_id)  # 404s if it is gone
    return _payload(state, project_id)


@router.patch("/{project_id}")
def update_project(
    project_id: str,
    payload: dict[str, Any] = Body(default={}),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """Rename, or replace the shared config. Both, in one call, if asked.

    ``name`` and ``config`` are independent and either may be absent -
    absent means "leave it alone", which is why the config is nested under
    its own key rather than being the body. A body that *was* the config
    could not express a rename without also claiming the config was empty.
    """
    state.project(project_id)

    try:
        if "name" in payload:
            state.projects.rename(project_id, str(payload.get("name") or ""))
        if "config" in payload:
            config = payload.get("config")
            if not isinstance(config, dict):
                raise Denied("config must be an object", status=422)
            state.projects.set_config(project_id, config)
    except ProjectError as exc:
        raise Denied(str(exc), status=422) from exc

    return _payload(state, project_id)


@router.delete("/{project_id}")
def delete_project(project_id: str, state: State = Depends(get_state)) -> dict[str, Any]:
    """Forget a project. **Nothing on disk is touched.**

    Same promise ``DELETE /datasets/{id}`` makes: the folders stay
    registered on the node and every image, caption and mask in them is
    exactly where it was. A student closing a project must never be a way
    to lose an afternoon of review work.
    """
    if not state.projects.delete(project_id):
        raise Denied(f"no project {project_id!r}", status=404)
    return {"id": project_id, "deleted": True, "datasets_kept": True}


@router.post("/{project_id}/datasets", status_code=201)
def add_dataset(
    project_id: str,
    payload: dict[str, Any] = Body(default={}),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """Put an already-registered dataset into a project."""
    state.project(project_id)
    dataset_id = str(payload.get("dataset") or "").strip()
    if not dataset_id:
        raise Denied("a dataset id is required", status=400)
    state.dataset(dataset_id)  # 404s rather than storing a dangling id
    state.projects.add_dataset(project_id, dataset_id)
    return _payload(state, project_id)


@router.delete("/{project_id}/datasets/{dataset_id}")
def remove_dataset(
    project_id: str,
    dataset_id: str,
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """Take a dataset out of a project. It stays registered on the node."""
    state.project(project_id)
    state.projects.remove_dataset(project_id, dataset_id)
    return _payload(state, project_id)


@router.get("/{project_id}/jobs")
def project_jobs(project_id: str, state: State = Depends(get_state)) -> dict[str, Any]:
    """This project's runs, and where each queued one sits in the whole queue.

    ``position`` counts against every project's jobs, not just this one's.
    Counting a filtered list would tell a student they were next while four
    other people were ahead of them, which is the one number this endpoint
    exists to get right.
    """
    state.project(project_id)
    jobs = [job for job in state.jobs.list() if job.project == project_id]
    return {
        "id": project_id,
        "jobs": [{**job.as_dict(), "position": state.jobs.position(job.id)} for job in jobs],
        "depth": state.jobs.depth(),
    }


def _payload(state: State, project_id: str) -> dict[str, Any]:
    project = state.project(project_id)
    resolved = state.project_datasets(project_id)
    return {
        **project.as_dict(),
        "datasets": [d.id for d in resolved],
        "dataset_details": [d.as_dict() for d in resolved],
        # What the store holds but the registry cannot resolve. Shown so a
        # missing folder is a visible fact rather than a row that silently
        # stopped appearing.
        "missing": [d for d in project.datasets if d not in {r.id for r in resolved}],
    }
