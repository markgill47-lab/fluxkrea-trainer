"""The event vocabulary and its emitter combinators."""

from __future__ import annotations

import json
import threading

import pytest

from fluxkrea.core import events


def test_events_serialise_with_a_kind() -> None:
    """Doc 06: events go straight onto the SSE stream."""
    payloads = [
        events.Progress(step=3, total=10, message="resizing").as_dict(),
        events.Log(line="hello", level="warning").as_dict(),
        events.LossPoint(step=42, value=0.31, image_id="punch_014").as_dict(),
        events.Finished(ok=False, detail="cancelled").as_dict(),
    ]
    assert [p["kind"] for p in payloads] == ["progress", "log", "loss", "finished"]
    for payload in payloads:
        json.dumps(payload)  # must not raise


def test_events_are_frozen() -> None:
    event = events.Log(line="x")
    with pytest.raises(Exception):
        event.line = "y"  # type: ignore[misc]


def test_progress_fraction_handles_indeterminate_and_clamps() -> None:
    assert events.Progress(0, 0).fraction == 0.0
    assert events.Progress(5, 10).fraction == 0.5
    assert events.Progress(20, 10).fraction == 1.0


def test_collector_records_and_filters() -> None:
    collector = events.Collector()
    collector(events.Log(line="one"))
    collector(events.Log(line="bad", level="error"))
    collector(events.Progress(1, 2))
    collector(events.Finished(ok=True))

    assert len(collector) == 4
    assert collector.lines("error") == ["bad"]
    assert len(collector.of(events.Log)) == 2
    assert collector.finished is not None and collector.finished.ok


def test_safe_shields_the_operation_from_a_broken_listener() -> None:
    def explode(event: events.Event) -> None:
        raise RuntimeError("SSE client went away")

    emit = events.safe(explode)
    emit(events.Log(line="this must not raise"))


def test_safe_normalises_none() -> None:
    assert events.safe(None) is events.no_op
    events.safe(None)(events.Log(line="x"))


def test_fanout_delivers_to_all_and_survives_one_failure() -> None:
    good = events.Collector()

    def explode(event: events.Event) -> None:
        raise RuntimeError("nope")

    emit = events.fanout(explode, good)
    emit(events.Log(line="delivered"))
    assert good.lines() == ["delivered"]


def test_prefixed_only_touches_logs() -> None:
    collector = events.Collector()
    emit = events.prefixed(collector, "[mask] ")
    emit(events.Log(line="detecting"))
    emit(events.Progress(1, 2, message="detecting"))

    assert collector.lines() == ["[mask] detecting"]
    assert collector.of(events.Progress)[0].message == "detecting"


def test_throttled_drops_intermediate_progress_only() -> None:
    collector = events.Collector()
    emit = events.throttled(collector, min_interval=60.0)

    emit(events.Progress(0, 100))  # first is always sent
    for step in range(1, 50):
        emit(events.Progress(step, 100))
    emit(events.Log(line="never dropped"))
    emit(events.Progress(100, 100))  # last is always sent

    progress = collector.of(events.Progress)
    assert [p.step for p in progress] == [0, 100]
    assert collector.lines() == ["never dropped"]


def test_iter_with_progress_brackets_the_work() -> None:
    collector = events.Collector()
    seen = list(events.iter_with_progress("abc", collector, "captioning"))

    assert seen == ["a", "b", "c"]
    steps = [p.step for p in collector.of(events.Progress)]
    assert steps == [0, 1, 2, 3]


def test_iter_with_progress_stops_when_cancelled() -> None:
    collector = events.Collector()
    cancel = threading.Event()
    seen = []

    for item in events.iter_with_progress(range(10), collector, "resizing", cancel):
        seen.append(item)
        if item == 2:
            cancel.set()

    assert seen == [0, 1, 2]
    assert any("Cancelled" in line for line in collector.lines("warning"))


def test_cancellation_helpers() -> None:
    cancel = threading.Event()
    assert not events.is_cancelled(None)
    assert not events.is_cancelled(cancel)
    cancel.set()
    assert events.is_cancelled(cancel)
    with pytest.raises(events.Cancelled, match="rename cancelled"):
        events.check_cancelled(cancel, "rename")


def test_collector_is_thread_safe() -> None:
    collector = events.Collector()

    def spam() -> None:
        for index in range(200):
            collector(events.Log(line=str(index)))

    threads = [threading.Thread(target=spam) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(collector) == 800
