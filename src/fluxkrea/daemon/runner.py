"""What the job queue actually runs.

The queue is deliberately ignorant of training: it holds specs, hands them
to a callable, and manages the lifecycle around it. This is that callable,
and it is the only place where a node's own facts - where its datasets
live, where ai-toolkit is checked out - meet a spec that was written
somewhere else.

That resolution step matters more than it looks. A spec submitted from a
laptop names a dataset by id, because the laptop's path for it means
nothing here. The node turns the id into its own path, and if the run is
masked, points ``mask_path`` at that copy's ``masks/`` folder rather than
at whatever the submitter happened to type.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core import paths
from ..core.backends import BackendError, TrainingBackend, supported_by
from ..core.dataset import validate
from ..core.events import Emitter, Log, safe
from ..core.backends.spec import RunSpec

if TYPE_CHECKING:  # pragma: no cover
    from .queue import Job
    from .state import State


@dataclass
class TrainResult:
    """What a finished job reports. Shaped like every other result type."""

    job_id: str
    model: str
    backend: str
    config_path: str = ""
    output: str = ""
    steps: int = 0
    ok: bool = True
    error: str = ""

    def summary(self) -> str:
        if not self.ok:
            return f"training failed: {self.error}"
        return f"{self.model} trained for {self.steps} steps"

    def as_dict(self) -> dict[str, Any]:
        return {
            "job": self.job_id,
            "model": self.model,
            "backend": self.backend,
            "config_path": self.config_path,
            "output": self.output,
            "steps": self.steps,
            "ok": self.ok,
            "error": self.error,
        }


def make_job_runner(state: State):  # noqa: ANN201 - returns a JobRunner
    """Build the callable the queue runs, bound to this node's state."""

    def run(job: Job, emit: Emitter, cancel: threading.Event) -> TrainResult:
        emit = safe(emit)
        spec = resolve(state, job.spec, emit)

        backend = supported_by(spec.model)
        if backend is None:
            raise BackendError(
                f"no backend handles model {spec.model!r}. "
                "v1 fell through to kohya for anything unrecognised."
            )

        _preflight(state, spec, emit)

        config_path = backend.generate_config(spec)
        job.config_path = str(config_path)
        emit(Log(line=f"Config written to {config_path}"))

        backend.start(config_path, emit, cancel, total_steps=spec.steps)

        return TrainResult(
            job_id=job.id,
            model=spec.model,
            backend=backend.name,
            config_path=str(config_path),
            output=spec.output,
            steps=job.progress.get("step", 0),
        )

    return run


def resolve(state: State, spec: RunSpec, emit: Emitter = None) -> RunSpec:
    """Turn a submitted spec into one this node can act on.

    The dataset becomes a local path, and a masked run gets the mask folder
    belonging to *this* copy of the dataset.
    """
    emit = safe(emit)
    dataset = state.registry.get(spec.dataset)
    if dataset is None:
        dataset = state.registry.by_path(spec.dataset)

    if dataset is None:
        candidate = paths.expand(spec.dataset)
        if not candidate.is_dir():
            raise BackendError(
                f"no dataset {spec.dataset!r} on this node, and no such folder. "
                "Push it first, or register the path."
            )
        folder = candidate
    else:
        folder = dataset.path

    resolved = replace(spec, dataset=folder.as_posix())

    if spec.mask_path:
        # Always the node's own masks folder, whatever the submitter typed.
        masks = paths.masks_dir(folder)
        resolved = replace(resolved, mask_path=masks.as_posix())
    elif spec.extra.get("masked"):
        resolved = replace(resolved, mask_path=paths.masks_dir(folder).as_posix())

    if not resolved.output:
        name = resolved.name or f"{resolved.model}-{folder.name}"
        root = state.config.backends.output_root or paths.runs_dir()
        resolved = replace(resolved, output=(paths.expand(root) / name).as_posix())

    return resolved


def _preflight(state: State, spec: RunSpec, emit: Emitter) -> None:
    """Check the dataset before spending an hour finding out it was wrong.

    A masked run is validated with ``require_masks``, because ai-toolkit
    trains an image with no matching mask unmasked and says nothing - which
    is the exact failure loss masking exists to prevent (doc 04).
    """
    folder = Path(spec.dataset)
    report = validate(
        folder,
        min_resolution=state.config.dataset.min_resolution,
        require_masks=bool(spec.mask_path),
        extensions=state.config.dataset.image_extensions,
        caption_ext=state.config.dataset.caption_ext,
    )

    emit(Log(line=f"Dataset: {report.summary()}"))
    for problem in report.warnings[:10]:
        emit(Log(line=f"  {problem}", level="warning"))

    if not report.ok:
        for problem in report.errors[:20]:
            emit(Log(line=f"  {problem}", level="error"))
        raise BackendError(
            f"{folder.name} has {len(report.errors)} validation errors; "
            "fix them or the run will train something you did not mean to"
        )

    if spec.mask_path and not paths.expand(spec.mask_path).is_dir():
        raise BackendError(
            f"the run is masked but {spec.mask_path} is not there. "
            "Run `fk dataset mask` on this node first."
        )


def attach(state: State) -> TrainingBackend | None:
    """Register the configured backends on this node and wire up the queue.

    Called once at daemon start. Registering a *configured* instance
    replaces the import-time default, which knows the models but not where
    anything lives on this machine.
    """
    from ..core.backends import register
    from ..core.backends.aitoolkit import AIToolkitBackend

    backend = AIToolkitBackend.from_config(state.config)
    register(backend)
    state.attach_job_runner(make_job_runner(state))
    return backend
