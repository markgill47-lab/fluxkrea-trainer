"""Training jobs.

The queue and the endpoint contract land in P3; the backends that actually
run a job land in P4. Until one is registered, ``POST /jobs`` accepts and
queues the spec but nothing picks it up, and ``GET /jobs`` says so - which
is more useful than a 501, because it lets the client, the queue
persistence and the event plumbing all be exercised end to end first.

The dispatch rule from doc 01 is enforced here rather than left to a
backend: **an unrecognised model raises.** v1's ``detect_backend`` falls
through to ``return 'kohya'``, so an unknown model silently routes to a
trainer that cannot handle it.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, Header, Query
from sse_starlette.sse import EventSourceResponse

from ..queue import QUEUED, RunSpec
from ..security import Denied
from ..state import State
from ..stream import parse_since, sse
from .deps import get_state

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("")
def list_jobs(
    status: str | None = Query(default=None),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    return {
        "jobs": [j.as_dict() for j in state.jobs.list(status=status)],
        "depth": state.jobs.depth(),
        "devices": state.jobs.devices,
        "runner": state.jobs.runner is not None,
    }


@router.post("", status_code=202)
def submit(payload: dict[str, Any] = Body(...), state: State = Depends(get_state)) -> dict[str, Any]:
    try:
        spec = RunSpec.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise Denied(str(exc), status=400) from exc

    if spec.device >= state.jobs.devices:
        raise Denied(
            f"this node has {state.jobs.devices} queue slot(s); device {spec.device} is not one",
            status=422,
        )

    if state.jobs.runner is not None:
        _check_model(state, spec)

    job = state.jobs.submit(spec)
    response = job.as_dict()
    if state.jobs.runner is None:
        response["warning"] = (
            "queued, but no training backend is registered on this node yet, "
            "so nothing will pick it up"
        )
    return response


@router.get("/{job_id}")
def get_job(job_id: str, state: State = Depends(get_state)) -> dict[str, Any]:
    return _job(state, job_id).as_dict()


@router.delete("/{job_id}")
def cancel(job_id: str, state: State = Depends(get_state)) -> dict[str, Any]:
    job = _job(state, job_id)
    was = job.status
    result = state.jobs.cancel(job_id)
    if result is None:
        raise Denied(f"job {job_id} is already {was}", status=409)
    return {"id": job_id, "was": was, "status": job.status, "dequeued": was == QUEUED}


@router.get("/{job_id}/logs")
def logs(
    job_id: str,
    since: int = Query(default=-1),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    job = _job(state, job_id)
    return {
        "id": job_id,
        "status": job.status,
        "events": [e.as_dict() for e in job.events_since(since)],
    }


@router.get("/{job_id}/events")
def events(
    job_id: str,
    since: int = Query(default=-1),
    last_event_id: str | None = Header(default=None, alias="last-event-id"),
    state: State = Depends(get_state),
) -> EventSourceResponse:
    return sse(_job(state, job_id), parse_since(last_event_id, since))


@router.get("/{job_id}/loss")
def loss(job_id: str, state: State = Depends(get_state)) -> dict[str, Any]:
    """The loss series.

    EMA, trend and outliers are deliberately *not* computed by a backend:
    ``analytics/loss.py`` will derive them from this series above the
    backend line, so every backend gets them equally (doc 02). Until that
    module exists, the raw points are what there is.
    """
    job = _job(state, job_id)
    return {
        "id": job_id,
        "points": [{"step": step, "value": value} for step, value in job.loss],
    }


def _check_model(state: State, spec: RunSpec) -> None:
    """Explicit dispatch. An unknown model raises rather than falling through."""
    from ...core.backends import supported_by

    backend = supported_by(spec.model)
    if backend is None:
        raise Denied(
            f"no backend handles model {spec.model!r}. "
            "v1 fell through to kohya for anything unrecognised, which is how a "
            "model silently reached a trainer that could not run it.",
            status=422,
        )


def _job(state: State, job_id: str):  # noqa: ANN202
    job = state.jobs.get(job_id)
    if job is None:
        raise Denied(f"no job {job_id!r}", status=404)
    return job
