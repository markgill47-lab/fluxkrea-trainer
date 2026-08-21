"""Task status, logs and the SSE stream.

Dataset operations are long-running too, and get the same treatment as
jobs: ``GET /tasks/{id}`` and ``GET /tasks/{id}/events`` (doc 06).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Query
from sse_starlette.sse import EventSourceResponse

from ..security import Denied
from ..state import State
from ..stream import parse_since, sse
from .deps import get_state

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("")
def list_tasks(
    dataset: str | None = Query(default=None),
    active: bool = Query(default=False),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    return {"tasks": [t.as_dict() for t in state.tasks.list(dataset_id=dataset, active_only=active)]}


@router.get("/{task_id}")
def get_task(task_id: str, state: State = Depends(get_state)) -> dict[str, Any]:
    return _task(state, task_id).as_dict()


@router.delete("/{task_id}")
def cancel_task(task_id: str, state: State = Depends(get_state)) -> dict[str, Any]:
    task = _task(state, task_id)
    cancelled = state.tasks.cancel(task_id)
    return {"id": task_id, "cancelling": cancelled, "status": task.status}


@router.get("/{task_id}/logs")
def logs(
    task_id: str,
    since: int = Query(default=-1),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """Backfill, for a client that reconnected or wants the whole record."""
    task = _task(state, task_id)
    return {
        "id": task_id,
        "status": task.status,
        "events": [e.as_dict() for e in task.events_since(since)],
    }


@router.get("/{task_id}/events")
def events(
    task_id: str,
    since: int = Query(default=-1),
    last_event_id: str | None = Header(default=None, alias="last-event-id"),
    state: State = Depends(get_state),
) -> EventSourceResponse:
    """The live stream. Resumes from ``Last-Event-ID`` if the browser sends one."""
    task = _task(state, task_id)
    return sse(task, parse_since(last_event_id, since))


def _task(state: State, task_id: str):  # noqa: ANN202
    task = state.tasks.get(task_id)
    if task is None:
        raise Denied(f"no task {task_id!r}", status=404)
    return task
