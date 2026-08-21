"""Typed events, replacing v1's bundle of injected callbacks.

v1 threads ``update_status_callback``, ``update_training_output_callback``,
``show_error_callback`` and ``show_info_callback`` through every manager
constructor: an event system with no registry, no typing, and no way for
two listeners to observe the same event - which is exactly what a daemon
streaming to several remote clients needs (doc 01).

Here there is one vocabulary. A core operation takes ``emit`` and calls
it; the CLI passes a printer, the daemon passes a fan-out to SSE
subscribers, a test passes :class:`Collector`. Core never knows which.

Events serialise straight onto the SSE stream via :meth:`Event.as_dict`,
so the dataclass a backend emits is what arrives on the laptop (doc 06).

Nothing in this module imports a UI toolkit or an HTTP framework, and it
must stay that way.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, fields
from typing import Any, Literal

Level = Literal["debug", "info", "warning", "error"]


# --------------------------------------------------------------------------
# event types
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Event:
    """Base for every event. ``kind`` is the SSE event name."""

    kind: str = field(init=False, default="event")

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready dict, including ``kind``.

        Explicit rather than ``dataclasses.asdict`` so that ``kind`` - an
        ``init=False`` class-level field - is always present, and so a
        field holding a ``Path`` serialises as a string rather than
        exploding at the JSON encoder.
        """
        out: dict[str, Any] = {"kind": self.kind}
        for f in fields(self):
            if f.name == "kind":
                continue
            out[f.name] = _plain(getattr(self, f.name))
        return out


def _plain(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return str(value)


@dataclass(frozen=True, slots=True)
class Progress(Event):
    """Position within a bounded operation.

    ``total`` of 0 means indeterminate - the operation is running but
    cannot say how long it will take. Clients render that as a bar with no
    fill rather than dividing by zero.
    """

    step: int
    total: int
    message: str = ""

    kind: str = field(init=False, default="progress")

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, max(0.0, self.step / self.total))


@dataclass(frozen=True, slots=True)
class Log(Event):
    """One line of output. ``level`` drives styling, not filtering."""

    line: str
    level: Level = "info"

    kind: str = field(init=False, default="log")


@dataclass(frozen=True, slots=True)
class LossPoint(Event):
    """One training loss sample.

    Deliberately dumb: backends emit points, and ``analytics/loss.py``
    computes EMA, trend and outliers from the stream for every backend
    equally. That is the fix for Klein having five analytics features the
    ai-toolkit backends silently lack (doc 02).
    """

    step: int
    value: float
    image_id: str | None = None

    kind: str = field(init=False, default="loss")


@dataclass(frozen=True, slots=True)
class Finished(Event):
    """Terminal event. Exactly one per operation, success or failure."""

    ok: bool
    detail: str = ""

    kind: str = field(init=False, default="finished")


#: What a core operation is handed. Synchronous, never raises through to
#: the caller (see :func:`safe`), and must be cheap - it is called from
#: inside tight loops.
Emitter = Callable[[Event], None]


def no_op(event: Event) -> None:
    """The default emitter. Core operations run headless with no listener."""


# --------------------------------------------------------------------------
# emitter combinators
# --------------------------------------------------------------------------


def fanout(*emitters: Emitter) -> Emitter:
    """Broadcast to several emitters. One failing listener never stops the rest."""

    live = [e for e in emitters if e is not no_op]
    if not live:
        return no_op

    def _emit(event: Event) -> None:
        for target in live:
            try:
                target(event)
            except Exception:  # noqa: BLE001 - a listener must never break the operation
                pass

    return _emit


def safe(emit: Emitter | None) -> Emitter:
    """Normalise ``None`` to a no-op and shield the caller from listener errors.

    Core operations call this once at entry, then emit freely. An emitter
    that raises - a closed SSE connection, a full queue - must not abort a
    rename halfway through a dataset.
    """
    if emit is None or emit is no_op:
        return no_op

    def _emit(event: Event) -> None:
        try:
            emit(event)
        except Exception:  # noqa: BLE001
            pass

    return _emit


def prefixed(emit: Emitter, prefix: str) -> Emitter:
    """Tag log lines from a sub-operation, leaving other events untouched."""
    inner = safe(emit)

    def _emit(event: Event) -> None:
        if isinstance(event, Log):
            inner(Log(line=f"{prefix}{event.line}", level=event.level))
        else:
            inner(event)

    return _emit


def throttled(emit: Emitter, min_interval: float = 0.1) -> Emitter:
    """Drop intermediate :class:`Progress` events closer than *min_interval*.

    Progress from a 4,000-image resize would otherwise push 4,000 SSE
    frames at a browser that can paint sixty. Only ``Progress`` is
    throttled: dropping a log line or a loss point loses data, dropping a
    progress tick loses nothing, since the next one supersedes it.
    """
    inner = safe(emit)
    state = {"last": 0.0}
    lock = threading.Lock()

    def _emit(event: Event) -> None:
        if isinstance(event, Progress):
            now = time.monotonic()
            boundary = event.step in (0, event.total)
            with lock:
                if not boundary and now - state["last"] < min_interval:
                    return
                # Boundaries reset the clock too: having just delivered one,
                # the next tick is no more urgent than any other.
                state["last"] = now
        inner(event)

    return _emit


class Collector:
    """An emitter that records. The test harness's listener.

    Thread-safe, because core operations are run on daemon worker threads
    and a test may assert from the main one.
    """

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._lock = threading.Lock()

    def __call__(self, event: Event) -> None:
        with self._lock:
            self._events.append(event)

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    @property
    def events(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def of(self, *types: type[Event]) -> list[Event]:
        return [e for e in self.events if isinstance(e, types)]

    def lines(self, *levels: Level) -> list[str]:
        wanted = set(levels)
        return [e.line for e in self.events if isinstance(e, Log) and (not wanted or e.level in wanted)]

    @property
    def finished(self) -> Finished | None:
        done = [e for e in self.events if isinstance(e, Finished)]
        return done[-1] if done else None

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


# --------------------------------------------------------------------------
# cancellation
# --------------------------------------------------------------------------


class Cancelled(Exception):
    """Raised when a cancel token is set and an operation cannot continue.

    Most operations check :func:`is_cancelled` at the top of their loop and
    return a partial result instead of raising; this exists for the ones
    that cannot leave a coherent partial state behind.
    """


def is_cancelled(cancel: threading.Event | None) -> bool:
    """Cancellation is a ``threading.Event`` passed in and checked at loop tops.

    Not ``progress.wasCanceled()`` read off a modal dialog, which is what
    made v1's processing loops untestable and unrunnable headless.
    """
    return cancel is not None and cancel.is_set()


def check_cancelled(cancel: threading.Event | None, what: str = "operation") -> None:
    if is_cancelled(cancel):
        raise Cancelled(f"{what} cancelled")


def iter_with_progress(
    items: Iterable[Any],
    emit: Emitter,
    message: str = "",
    cancel: threading.Event | None = None,
) -> Iterator[Any]:
    """Iterate a sized collection, emitting progress and honouring cancellation.

    Stops cleanly when cancelled - the caller sees a short iteration and
    decides what a partial result means, rather than an exception unwinding
    through a half-written dataset.
    """
    seq = list(items)
    total = len(seq)
    emit(Progress(step=0, total=total, message=message))
    for index, item in enumerate(seq, start=1):
        if is_cancelled(cancel):
            emit(Log(line=f"Cancelled after {index - 1} of {total}", level="warning"))
            return
        yield item
        emit(Progress(step=index, total=total, message=message))
