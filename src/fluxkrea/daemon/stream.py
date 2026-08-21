"""Server-Sent Events.

Events are strictly server-to-client, so SSE fits better than WebSockets:
it survives proxies, reconnects natively with ``Last-Event-ID``, needs no
extra dependency on the client side, and can be read with plain ``curl``
while debugging (doc 06).

The core's event types serialise directly onto the stream - the same
dataclass a training loop emits is what arrives on the laptop.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, Protocol

from sse_starlette.sse import EventSourceResponse

#: How often to send a comment frame when nothing is happening. Keeps
#: intermediaries from timing out an idle training run.
KEEPALIVE = 15.0

#: Per-subscriber backlog. A client too slow to keep up loses frames rather
#: than growing the daemon's memory without bound - and can recover the gap
#: with ``since``, which is why the buffer on the task side exists.
BACKLOG = 2048


class Streamable(Protocol):
    """What both ``Task`` and ``Job`` provide."""

    def events_since(self, since: int = -1) -> list[Any]: ...
    def subscribe(self, queue: Any) -> None: ...
    def unsubscribe(self, queue: Any) -> None: ...
    @property
    def done(self) -> bool: ...


async def event_stream(source: Streamable, since: int = -1) -> AsyncIterator[dict[str, Any]]:
    """Backfill from *since*, then follow live until the source finishes.

    Subscribing happens **before** the backfill is read, so an event landing
    between the two is delivered by the live path rather than falling into
    the gap. Duplicates are filtered by index on the way out.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=BACKLOG)
    loop = asyncio.get_running_loop()

    class Bridge:
        """Hands an envelope from a worker thread to the event loop."""

        def put_nowait(self, envelope: Any) -> None:
            loop.call_soon_threadsafe(_offer, queue, envelope)

    bridge = Bridge()
    source.subscribe(bridge)
    delivered = since

    try:
        for envelope in source.events_since(since):
            if envelope.index > delivered:
                delivered = envelope.index
                yield frame(envelope)

        while True:
            if source.done and queue.empty():
                # Drain anything the terminal event raced past us, then stop.
                for envelope in source.events_since(delivered):
                    delivered = envelope.index
                    yield frame(envelope)
                return
            try:
                envelope = await asyncio.wait_for(queue.get(), timeout=1.0)
            except TimeoutError:
                continue
            if envelope.index > delivered:
                delivered = envelope.index
                yield frame(envelope)
    finally:
        source.unsubscribe(bridge)


def frame(envelope: Any) -> dict[str, Any]:
    """One SSE frame: named by event kind, with the index as its id."""
    payload = envelope.as_dict()
    return {
        "id": str(envelope.index),
        "event": payload.get("kind", "event"),
        "data": json.dumps(payload),
    }


def _offer(queue: asyncio.Queue, envelope: Any) -> None:
    try:
        queue.put_nowait(envelope)
    except asyncio.QueueFull:
        # Dropping is correct here: the client can ask for the gap by index.
        pass


def sse(source: Streamable, since: int = -1) -> EventSourceResponse:
    return EventSourceResponse(event_stream(source, since), ping=int(KEEPALIVE))


def parse_since(last_event_id: str | None, since: int | None) -> int:
    """``Last-Event-ID`` wins over an explicit ``since``; a browser sends it."""
    for candidate in (last_event_id, since):
        if candidate is None or candidate == "":
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return -1
