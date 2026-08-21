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
from pathlib import Path

from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from ...core.backends.plan import plan, rate_from_history
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


@router.post("/plan")
def plan_run(
    payload: dict[str, Any] = Body(default={}),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """How many steps a run would be, and how long it would take.

    v1's "Training steps are calculated automatically" box, as an endpoint,
    so the CLI and the browser agree on the arithmetic and on where the
    time estimate came from. Counting the images is the expensive half, and
    it is also the half that goes stale - which is why this is a call
    rather than a number the client caches.
    """
    dataset_id = str(payload.get("dataset") or "").strip()
    if not dataset_id:
        raise Denied("a plan needs a dataset", status=400)

    images = len(state.items(dataset_id))
    history = [job.as_dict() for job in state.jobs.list()]
    rate, basis = rate_from_history(history, str(payload.get("model") or ""))

    computed = plan(
        images,
        repeats=int(payload.get("repeats") or 1),
        epochs=int(payload.get("epochs") or 1),
        seconds_per_step=rate,
        basis=basis,
    )
    return {"dataset": dataset_id, **computed.as_dict()}


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
def loss(
    job_id: str,
    points: int = Query(default=2000, description="cap on returned points; 0 for all"),
    ema: int = Query(default=50, description="EMA window"),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """The loss series, with EMA, trend and outliers.

    None of this is computed by a backend. Backends emit ``LossPoint``
    events and ``analytics/loss.py`` derives the rest, so every backend
    gets the same features rather than whichever ones its author happened
    to implement (doc 02).

    Decimation happens here rather than in the client, so a 20,000-step run
    also does not send 20,000 points across a tunnel to draw 2,000 pixels.
    """
    job = _job(state, job_id)
    return {"id": job_id, **job.series.as_dict(decimate_to=points, ema_window=ema)}


@router.get("/{job_id}/samples")
def samples(job_id: str, state: State = Depends(get_state)) -> dict[str, Any]:
    """Sample images generated during the run, newest first.

    Served as a listing of steps and URLs rather than bytes: doc 10 warns
    that a run producing a 1024x1024 sample every 400 steps fills a strip
    fast, so they are thumbnails in the strip and full size only on click.
    """
    job = _job(state, job_id)
    return {"id": job_id, "samples": _find_samples(job)}


@router.get("/{job_id}/samples/{name}")
def sample_image(job_id: str, name: str, state: State = Depends(get_state)) -> FileResponse:
    job = _job(state, job_id)
    for entry in _find_samples(job):
        if entry["name"] == name:
            return FileResponse(entry["path"], media_type="image/jpeg")
    raise Denied(f"no sample {name!r} for job {job_id}", status=404)


@router.get("/{job_id}/folders")
def folders(job_id: str, state: State = Depends(get_state)) -> dict[str, Any]:
    """Where this run wrote, so a client can show or open it.

    ``exists`` is checked rather than assumed: the samples folder does not
    appear until the first sample is rendered, and offering to open a
    folder that is not there yet is worse than not offering.
    """
    job = _job(state, job_id)
    from ...core import paths as core_paths

    output = core_paths.expand(job.spec.output) if job.spec.output else None
    samples = _samples_dir(job)

    return {
        "id": job_id,
        "output": output.as_posix() if output else "",
        "output_exists": bool(output and output.is_dir()),
        "samples": samples.as_posix() if samples else "",
        "samples_exists": bool(samples and samples.is_dir()),
    }


@router.post("/{job_id}/folders/open")
def open_run_folder(
    job_id: str,
    payload: dict[str, Any] = Body(default={}),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """Open the run's output or samples folder in the node's file manager.

    Returns 200 with ``opened: false`` and a reason rather than an error
    status: a headless node cannot open anything, and that is an answer to
    the question, not a failure. The path comes back either way, because
    over a tunnel it is the only useful half.
    """
    from ..reveal import open_folder

    job = _job(state, job_id)
    which = str(payload.get("which") or "samples").strip().lower()
    if which not in ("samples", "output"):
        raise Denied(f"unknown folder {which!r}; expected 'samples' or 'output'", status=400)

    from ...core import paths as core_paths

    if which == "samples":
        target = _samples_dir(job)
    else:
        target = core_paths.expand(job.spec.output) if job.spec.output else None

    if target is None:
        raise Denied(f"job {job_id} has no {which} folder recorded", status=404)

    opened, why = open_folder(target)
    return {"id": job_id, "which": which, "path": target.as_posix(), "opened": opened, "detail": why}


def _samples_dir(job: Any) -> Path | None:
    """The samples folder for a run, whether or not it exists yet."""
    from ...core import paths as core_paths

    if not job.spec.output:
        return None
    root = core_paths.expand(job.spec.output)
    for name in SAMPLE_DIRS:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    # Not there yet - name the one the trainer will create.
    return root / SAMPLE_DIRS[0]


#: Where trainers write samples, relative to the run's output folder.
SAMPLE_DIRS = ("samples", "sample")
SAMPLE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _find_samples(job: Any) -> list[dict[str, Any]]:
    """Locate sample images in a run's output folder.

    The step is parsed from the filename, which is how every trainer here
    names them. A file that does not carry one still gets listed, ordered
    by mtime, rather than being hidden because it did not match a pattern.
    """
    import re

    from ...core import paths as core_paths

    root = core_paths.expand(job.spec.output) if job.spec.output else None
    if root is None or not root.is_dir():
        return []

    found: list[dict[str, Any]] = []
    for folder in (root, *(root / name for name in SAMPLE_DIRS)):
        if not folder.is_dir():
            continue
        for entry in folder.iterdir():
            if not entry.is_file() or entry.suffix.lower() not in SAMPLE_SUFFIXES:
                continue
            digits = re.findall(r"\d+", entry.stem)
            step = int(digits[-1]) if digits else 0
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            found.append(
                {
                    "name": entry.name,
                    "step": step,
                    "mtime": mtime,
                    "path": str(entry),
                    "url": f"/api/v1/jobs/{job.id}/samples/{entry.name}",
                }
            )

    found.sort(key=lambda entry: (entry["step"], entry["mtime"]))
    return found


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
