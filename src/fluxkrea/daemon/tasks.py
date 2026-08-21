"""Long-running operations, run on worker threads and streamed.

Core operations are synchronous and blocking, and say so (doc 02). This is
the layer that makes that fact useful: a task owns one operation, runs it
on a worker thread, buffers every event it emits, and fans those events out
to any number of subscribers.

Three properties matter and are easy to lose:

**Every event is buffered, in order, with an index.** A client that
reconnects passes ``since`` and gets the gap - the same job as SSE's
``Last-Event-ID``. Without the buffer, closing a laptop lid loses the
middle of a run.

**Multiple subscribers see the same stream.** v1's callback bundle has no
registry and no way for two listeners to observe one event, which is
exactly what a daemon streaming to several remote clients needs.

**The runner emits ``Finished``, exactly once.** Core operations
deliberately do not (see ``core/dataset/ops/__init__.py``), so composing
them cannot produce two terminal events.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ..core.events import Emitter, Event, Finished, Log, Progress, safe

#: Events kept per task for reconnect backfill. A dataset operation over a
#: few thousand images emits a few thousand events; this is generous.
BUFFER = 20_000

#: Finished tasks kept before the oldest is dropped.
HISTORY = 200

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
TERMINAL = frozenset({DONE, FAILED, CANCELLED})


@dataclass(frozen=True, slots=True)
class Envelope:
    """One event with the index a reconnecting client resumes from."""

    index: int
    event: Event

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, **self.event.as_dict()}


@dataclass
class Task:
    """One running operation and everything a client can ask about it."""

    id: str
    kind: str
    dataset_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    status: str = QUEUED
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: Any = None
    error: str = ""

    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _events: deque[Envelope] = field(default_factory=lambda: deque(maxlen=BUFFER), repr=False)
    _subscribers: list[Any] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _next: int = 0

    # -- event plumbing ---------------------------------------------------

    def emit(self, event: Event) -> None:
        """Buffer an event and hand it to every live subscriber."""
        with self._lock:
            envelope = Envelope(index=self._next, event=event)
            self._next += 1
            self._events.append(envelope)
            targets = list(self._subscribers)
            if isinstance(event, Progress):
                self.detail["progress"] = {"step": event.step, "total": event.total}
        for queue in targets:
            try:
                queue.put_nowait(envelope)
            except Exception:  # noqa: BLE001 - a full or closed queue is the client's problem
                pass

    def as_emitter(self) -> Emitter:
        return safe(self.emit)

    def events_since(self, since: int = -1) -> list[Envelope]:
        with self._lock:
            return [e for e in self._events if e.index > since]

    def subscribe(self, queue: Any) -> None:
        with self._lock:
            self._subscribers.append(queue)

    def unsubscribe(self, queue: Any) -> None:
        with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    # -- lifecycle --------------------------------------------------------

    @property
    def done(self) -> bool:
        return self.status in TERMINAL

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the task finishes. For tests and synchronous clients."""
        return self._done.wait(timeout)

    def request_cancel(self) -> None:
        self.cancel.set()

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "dataset": self.dataset_id,
            "status": self.status,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "detail": self.detail,
            "error": self.error,
            "events": self._next,
        }
        if self.result is not None:
            payload["result"] = (
                self.result.as_dict() if hasattr(self.result, "as_dict") else self.result
            )
        return payload


class TaskRunner:
    """Owns the worker threads and the task history."""

    def __init__(self, workers: int = 2) -> None:
        self.workers = max(1, workers)
        self._tasks: dict[str, Task] = {}
        self._order: deque[str] = deque()
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._slots = threading.BoundedSemaphore(self.workers)

    # -- submission -------------------------------------------------------

    def submit(
        self,
        kind: str,
        work: Callable[[Emitter, threading.Event], Any],
        *,
        dataset_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> Task:
        """Start *work* on a worker thread. Returns immediately.

        *work* receives the emitter and the cancel token, and returns
        whatever the operation returns - typically a result dataclass with
        ``as_dict``, which is what the task endpoint serves.
        """
        task = Task(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            dataset_id=dataset_id,
            detail=dict(detail or {}),
        )
        with self._lock:
            self._tasks[task.id] = task
            self._order.append(task.id)
            self._trim()

        thread = threading.Thread(target=self._run, args=(task, work), name=f"task-{task.id}", daemon=True)
        self._threads[task.id] = thread
        thread.start()
        return task

    def _run(self, task: Task, work: Callable[[Emitter, threading.Event], Any]) -> None:
        # Bound concurrency without an executor, so a queued task still has a
        # real Task object a client can poll from the instant it is submitted.
        self._slots.acquire()
        detail = ""
        try:
            task.status = RUNNING
            task.started = time.time()
            result = work(task.as_emitter(), task.cancel)
            task.result = result

            if task.cancel.is_set():
                task.status = CANCELLED
                detail = "cancelled"
            else:
                ok = bool(getattr(result, "ok", True))
                task.status = DONE if ok else FAILED
                detail = result.summary() if hasattr(result, "summary") else ""
                if not ok:
                    task.error = detail
        except Exception as exc:  # noqa: BLE001 - a task must never kill the daemon
            task.status = FAILED
            task.error = f"{type(exc).__name__}: {exc}"
            detail = task.error
            task.emit(Log(line=task.error, level="error"))
            task.emit(Log(line=traceback.format_exc(), level="debug"))
        finally:
            task.finished = time.time()
            # The one terminal event, emitted by the runner. Core operations
            # never emit Finished, so there can only be this one.
            task.emit(Finished(ok=task.status == DONE, detail=detail))
            task._done.set()
            self._slots.release()

    # -- access -----------------------------------------------------------

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self, *, dataset_id: str | None = None, active_only: bool = False) -> list[Task]:
        with self._lock:
            tasks = [self._tasks[i] for i in self._order if i in self._tasks]
        if dataset_id:
            tasks = [t for t in tasks if t.dataset_id == dataset_id]
        if active_only:
            tasks = [t for t in tasks if not t.done]
        return sorted(tasks, key=lambda t: t.created, reverse=True)

    def cancel(self, task_id: str) -> bool:
        task = self.get(task_id)
        if task is None or task.done:
            return False
        task.request_cancel()
        return True

    def active(self) -> int:
        return sum(1 for t in self.list() if not t.done)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Ask every running task to stop, then wait briefly for them."""
        for task in self.list(active_only=True):
            task.request_cancel()
        deadline = time.monotonic() + timeout
        for thread in list(self._threads.values()):
            remaining = deadline - time.monotonic()
            if remaining > 0:
                thread.join(remaining)

    def _trim(self) -> None:
        while len(self._order) > HISTORY:
            oldest = self._order[0]
            task = self._tasks.get(oldest)
            if task is not None and not task.done:
                return  # never drop something still running
            self._order.popleft()
            self._tasks.pop(oldest, None)
            self._threads.pop(oldest, None)

    def __iter__(self) -> Iterator[Task]:
        return iter(self.list())
