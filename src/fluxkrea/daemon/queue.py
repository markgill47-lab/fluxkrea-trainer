"""The training job queue. One job per GPU at a time; the rest wait.

Persisted to disk, because the queue outliving the process is the whole
point: submitting work from a laptop and closing it must not be a way to
lose the work. Doc 06 is specific about the restart case - **in-flight
jobs are marked interrupted rather than silently vanishing**, so a reboot
during a training run leaves evidence instead of a gap.

Multi-GPU nodes get one slot per device, with the device pinned in the
``RunSpec``. Placement is explicit (``fk train --node olympus-2``); there
is no automatic scheduler, and a coordinator would be the price of one.

The backends themselves arrive in P4. This module is deliberately agnostic
about what a job *does*: it holds a spec, hands it to a runner callable,
and manages the lifecycle around it.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core import paths
from ..core.backends.spec import RunSpec
from ..core.events import Emitter, Event, Finished, Log, LossPoint, Progress, safe

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
#: A job that was running when the daemon stopped. Never silently dropped.
INTERRUPTED = "interrupted"

TERMINAL = frozenset({DONE, FAILED, CANCELLED, INTERRUPTED})

#: Events kept in memory per job for reconnect backfill. Training runs are
#: long and chatty; the on-disk log is the durable record.
BUFFER = 50_000


@dataclass
class Job:
    id: str
    spec: RunSpec
    status: str = QUEUED
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    error: str = ""
    #: Path of the generated backend config, once a backend has rendered it.
    config_path: str = ""
    progress: dict[str, int] = field(default_factory=lambda: {"step": 0, "total": 0})

    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _events: deque[Any] = field(default_factory=lambda: deque(maxlen=BUFFER), repr=False)
    _subscribers: list[Any] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _next: int = 0
    #: Loss series, kept so ``GET /jobs/{id}/loss`` can answer after the fact.
    loss: list[tuple[int, float]] = field(default_factory=list, repr=False)

    @property
    def device(self) -> int:
        return self.spec.device

    @property
    def done(self) -> bool:
        return self.status in TERMINAL

    def emit(self, event: Event) -> None:
        from .tasks import Envelope

        with self._lock:
            envelope = Envelope(index=self._next, event=event)
            self._next += 1
            self._events.append(envelope)
            targets = list(self._subscribers)
            if isinstance(event, Progress):
                self.progress = {"step": event.step, "total": event.total}
        if isinstance(event, LossPoint):
            # Kept so GET /jobs/{id}/loss can answer after the run has ended.
            # analytics/loss.py derives EMA, trend and outliers from these,
            # above the backend line, so every backend gets them (doc 02).
            self.loss.append((event.step, event.value))
        for queue in targets:
            try:
                queue.put_nowait(envelope)
            except Exception:  # noqa: BLE001
                pass

    def as_emitter(self) -> Emitter:
        return safe(self.emit)

    def events_since(self, since: int = -1) -> list[Any]:
        with self._lock:
            return [e for e in self._events if e.index > since]

    def subscribe(self, queue: Any) -> None:
        with self._lock:
            self._subscribers.append(queue)

    def unsubscribe(self, queue: Any) -> None:
        with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "error": self.error,
            "device": self.device,
            "config_path": self.config_path,
            "progress": self.progress,
            "events": self._next,
            "spec": self.spec.as_dict(),
        }

    def persisted(self) -> dict[str, Any]:
        """The subset written to disk. Threads and buffers do not survive."""
        return {
            "id": self.id,
            "status": self.status,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "error": self.error,
            "config_path": self.config_path,
            "spec": self.spec.as_dict(),
        }

    @classmethod
    def restored(cls, data: dict[str, Any]) -> Job:
        job = cls(
            id=str(data["id"]),
            spec=RunSpec.from_dict(data["spec"]),
            status=str(data.get("status", QUEUED)),
            created=float(data.get("created", time.time())),
            started=data.get("started"),
            finished=data.get("finished"),
            error=str(data.get("error", "")),
            config_path=str(data.get("config_path", "")),
        )
        if job.status == RUNNING:
            # The daemon stopped while this was training. Say so rather than
            # letting it read as still running or quietly disappear.
            job.status = INTERRUPTED
            job.error = "the daemon stopped while this job was running"
            job.finished = time.time()
        if job.done:
            job._done.set()
        return job


#: What actually runs a job. Supplied by the backend layer in P4.
JobRunner = Callable[[Job, Emitter, threading.Event], Any]


class JobQueue:
    """One slot per device, persisted across restarts."""

    def __init__(
        self,
        runner: JobRunner | None = None,
        *,
        devices: int = 1,
        directory: Path | None = None,
    ) -> None:
        self.runner = runner
        self.devices = max(1, devices)
        self._dir = directory if directory is not None else paths.queue_dir()
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque()
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        self._wake = threading.Condition(self._lock)
        self._stopping = False
        self._load()

    # -- persistence ------------------------------------------------------

    @property
    def file(self) -> Path:
        return self._dir / "jobs.json"

    def _load(self) -> None:
        if not self.file.is_file():
            return
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for entry in data.get("jobs", []):
            try:
                job = Job.restored(entry)
            except (KeyError, TypeError, ValueError):
                continue
            self._jobs[job.id] = job
            self._order.append(job.id)

    def _save(self) -> None:
        paths.ensure_dir(self._dir)
        payload = {
            "version": 1,
            "jobs": [self._jobs[i].persisted() for i in self._order if i in self._jobs],
        }
        tmp = self.file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.file)

    # -- submission -------------------------------------------------------

    def submit(self, spec: RunSpec) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], spec=spec)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._save()
        self._pump()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, *, status: str | None = None) -> list[Job]:
        with self._lock:
            jobs = [self._jobs[i] for i in self._order if i in self._jobs]
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created, reverse=True)

    def depth(self) -> int:
        return len(self.list(status=QUEUED))

    def running(self, device: int | None = None) -> list[Job]:
        jobs = self.list(status=RUNNING)
        return [j for j in jobs if device is None or j.device == device] if jobs else []

    def cancel(self, job_id: str) -> str | None:
        """Cancel a running job, or dequeue one that has not started.

        Returns the resulting status, or ``None`` if there was no such job.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.done:
                return None
            if job.status == QUEUED:
                job.status = CANCELLED
                job.finished = time.time()
                job._done.set()
                self._save()
                job.emit(Finished(ok=False, detail="dequeued before it started"))
                return CANCELLED
            job.cancel.set()
        return RUNNING  # it will report CANCELLED once the backend stops

    # -- execution --------------------------------------------------------

    def _pump(self) -> None:
        """Start whatever can start. Called on submit and on completion."""
        if self.runner is None or self._stopping:
            return
        with self._lock:
            busy = {job.device for job in self.list(status=RUNNING)}
            for job in sorted(self.list(status=QUEUED), key=lambda j: j.created):
                if job.device in busy or job.device >= self.devices:
                    continue
                busy.add(job.device)
                job.status = RUNNING
                job.started = time.time()
                self._save()
                thread = threading.Thread(
                    target=self._run, args=(job,), name=f"job-{job.id}", daemon=True
                )
                self._threads[job.id] = thread
                thread.start()

    def _run(self, job: Job) -> None:
        assert self.runner is not None
        detail = ""
        try:
            result = self.runner(job, job.as_emitter(), job.cancel)
            if job.cancel.is_set():
                job.status = CANCELLED
                detail = "cancelled"
            else:
                ok = bool(getattr(result, "ok", True))
                job.status = DONE if ok else FAILED
                detail = result.summary() if hasattr(result, "summary") else ""
                if not ok:
                    job.error = detail
        except Exception as exc:  # noqa: BLE001 - one bad job must not stop the queue
            job.status = FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            detail = job.error
            job.emit(Log(line=job.error, level="error"))
            job.emit(Log(line=traceback.format_exc(), level="debug"))
        finally:
            job.finished = time.time()
            job.emit(Finished(ok=job.status == DONE, detail=detail))
            job._done.set()
            with self._lock:
                self._save()
            self._pump()

    def stop(self, timeout: float = 10.0) -> None:
        """Ask running jobs to stop and persist the queue as it stands."""
        self._stopping = True
        for job in self.list(status=RUNNING):
            job.cancel.set()
        deadline = time.monotonic() + timeout
        for thread in list(self._threads.values()):
            remaining = deadline - time.monotonic()
            if remaining > 0:
                thread.join(remaining)
        with self._lock:
            self._save()
