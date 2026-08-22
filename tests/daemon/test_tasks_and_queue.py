"""The task runner, the SSE stream, and the persistent job queue."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import httpx
import pytest

from fluxkrea.core.events import Log, LossPoint, Progress
from fluxkrea.daemon import queue as q
from fluxkrea.daemon.queue import INTERRUPTED, JobQueue, RunSpec
from fluxkrea.daemon.tasks import CANCELLED, DONE, FAILED, TaskRunner
from tests.daemon.conftest import register


class Result:
    def __init__(self, ok: bool = True, summary: str = "did the thing") -> None:
        self.ok = ok
        self._summary = summary

    def summary(self) -> str:
        return self._summary

    def as_dict(self) -> dict:
        return {"ok": self.ok, "summary": self._summary}


# --------------------------------------------------------------------------
# task runner
# --------------------------------------------------------------------------


def test_a_task_runs_and_reports(tmp_path: Path) -> None:
    runner = TaskRunner(workers=2)

    def work(emit, cancel):
        emit(Progress(1, 2, "working"))
        emit(Log(line="halfway"))
        emit(Progress(2, 2, "working"))
        return Result()

    task = runner.submit("test", work)
    assert task.wait(10)

    assert task.status == DONE
    payload = task.as_dict()
    assert payload["result"]["ok"] is True
    assert payload["detail"]["progress"] == {"step": 2, "total": 2}


def test_the_runner_emits_exactly_one_finished() -> None:
    """Core operations never emit Finished, so composing them cannot double it."""
    runner = TaskRunner()
    task = runner.submit("test", lambda emit, cancel: Result())
    task.wait(10)

    kinds = [e.event.kind for e in task.events_since(-1)]
    assert kinds.count("finished") == 1
    assert kinds[-1] == "finished"


def test_an_exception_fails_the_task_without_killing_the_daemon() -> None:
    runner = TaskRunner()

    def explode(emit, cancel):
        raise RuntimeError("something went wrong")

    task = runner.submit("test", explode)
    task.wait(10)

    assert task.status == FAILED
    assert "RuntimeError: something went wrong" in task.error
    assert task.events_since(-1)[-1].event.kind == "finished"

    # And the runner still works afterwards.
    assert runner.submit("test", lambda e, c: Result()).wait(10)


def test_a_task_can_be_cancelled() -> None:
    runner = TaskRunner()
    started = threading.Event()

    def slow(emit, cancel):
        started.set()
        while not cancel.is_set():
            time.sleep(0.01)
        return Result()

    task = runner.submit("test", slow)
    assert started.wait(5)
    assert runner.cancel(task.id)
    assert task.wait(10)
    assert task.status == CANCELLED


def test_events_are_indexed_for_reconnect() -> None:
    runner = TaskRunner()

    def chatty(emit, cancel):
        for index in range(5):
            emit(Log(line=f"line {index}"))
        return Result()

    task = runner.submit("test", chatty)
    task.wait(10)

    everything = task.events_since(-1)
    assert [e.index for e in everything] == list(range(len(everything)))
    assert [e.index for e in task.events_since(2)] == [i for i in range(len(everything)) if i > 2]


def test_two_subscribers_see_the_same_stream() -> None:
    """v1's callback bundle has no way for two listeners to observe one event."""
    import queue as stdqueue

    runner = TaskRunner()
    first: stdqueue.Queue = stdqueue.Queue()
    second: stdqueue.Queue = stdqueue.Queue()
    ready = threading.Event()

    def work(emit, cancel):
        ready.wait(5)
        emit(Log(line="broadcast"))
        return Result()

    task = runner.submit("test", work)
    task.subscribe(first)
    task.subscribe(second)
    ready.set()
    task.wait(10)

    assert not first.empty() and not second.empty()


def test_history_never_drops_a_running_task() -> None:
    runner = TaskRunner(workers=4)
    tasks = [runner.submit("test", lambda e, c: Result()) for _ in range(5)]
    for task in tasks:
        task.wait(10)
    assert len(runner.list()) == 5


# --------------------------------------------------------------------------
# SSE
# --------------------------------------------------------------------------


def test_sse_delivers_the_whole_stream(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    task_id = api.post(f"/datasets/{dataset_id}/ops/validate", json={}).json()["id"]

    frames = []
    with api.stream("GET", f"/tasks/{task_id}/events") as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data:"):
                frames.append(json.loads(line[5:].strip()))

    assert frames[-1]["kind"] == "finished"
    assert frames[-1]["ok"] is True
    assert [f["index"] for f in frames] == sorted(f["index"] for f in frames)


def test_sse_backfills_from_since(api: httpx.Client, dataset: Path) -> None:
    """A client that reconnected asks for the gap by index."""
    dataset_id = register(api, dataset)
    task_id = api.post(f"/datasets/{dataset_id}/ops/validate", json={}).json()["id"]
    api.app_state.tasks.get(task_id).wait(10)  # type: ignore[attr-defined]

    everything = api.get(f"/tasks/{task_id}/logs").json()["events"]
    tail = api.get(f"/tasks/{task_id}/logs", params={"since": 0}).json()["events"]

    assert len(everything) > len(tail)
    assert tail[0]["index"] == 1


def test_sse_resumes_from_last_event_id(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    task_id = api.post(f"/datasets/{dataset_id}/ops/validate", json={}).json()["id"]
    api.app_state.tasks.get(task_id).wait(10)  # type: ignore[attr-defined]

    frames = []
    with api.stream("GET", f"/tasks/{task_id}/events", headers={"last-event-id": "0"}) as response:
        for line in response.iter_lines():
            if line.startswith("data:"):
                frames.append(json.loads(line[5:].strip()))

    assert frames and frames[0]["index"] == 1


# --------------------------------------------------------------------------
# job queue
# --------------------------------------------------------------------------


def test_a_job_runs_and_streams(tmp_path: Path) -> None:
    def runner(job, emit, cancel):
        emit(Progress(1, 10, "training"))
        emit(LossPoint(step=1, value=0.42))
        return Result(summary="trained")

    jobs = JobQueue(runner=runner, devices=1, directory=tmp_path)
    job = jobs.submit(RunSpec(model="krea2", dataset="poses"))
    assert job.wait(10)

    assert job.status == q.DONE
    assert job.progress == {"step": 1, "total": 10}
    assert job.loss == [(1, 0.42)]


def test_one_job_per_device(tmp_path: Path) -> None:
    running = threading.Event()
    release = threading.Event()
    concurrent = []
    lock = threading.Lock()

    def runner(job, emit, cancel):
        with lock:
            concurrent.append(job.id)
            peak = len(concurrent)
        running.set()
        release.wait(5)
        with lock:
            concurrent.remove(job.id)
        return Result(summary=f"peak {peak}")

    jobs = JobQueue(runner=runner, devices=1, directory=tmp_path)
    first = jobs.submit(RunSpec(model="krea2", dataset="a", device=0))
    second = jobs.submit(RunSpec(model="krea2", dataset="b", device=0))

    assert running.wait(5)
    assert second.status == q.QUEUED, "a second job on the same GPU must wait"
    release.set()
    assert first.wait(10) and second.wait(10)


def test_separate_devices_run_together(tmp_path: Path) -> None:
    both = threading.Barrier(2, timeout=10)

    def runner(job, emit, cancel):
        both.wait()  # deadlocks unless the two really are concurrent
        return Result()

    jobs = JobQueue(runner=runner, devices=2, directory=tmp_path)
    a = jobs.submit(RunSpec(model="krea2", dataset="a", device=0))
    b = jobs.submit(RunSpec(model="krea2", dataset="b", device=1))

    assert a.wait(10) and b.wait(10)
    assert a.status == q.DONE and b.status == q.DONE


def test_the_queue_survives_a_restart(tmp_path: Path) -> None:
    jobs = JobQueue(runner=None, devices=1, directory=tmp_path)
    jobs.submit(RunSpec(model="krea2", dataset="poses"))

    restarted = JobQueue(runner=None, devices=1, directory=tmp_path)
    assert [j.status for j in restarted.list()] == [q.QUEUED]
    assert restarted.list()[0].spec.model == "krea2"


def test_an_in_flight_job_is_marked_interrupted_not_lost(tmp_path: Path) -> None:
    """Doc 06: a reboot mid-run must leave evidence, not a gap."""
    jobs = JobQueue(runner=None, devices=1, directory=tmp_path)
    job = jobs.submit(RunSpec(model="krea2", dataset="poses"))
    job.status = q.RUNNING
    jobs._save()

    restarted = JobQueue(runner=None, devices=1, directory=tmp_path)
    restored = restarted.get(job.id)
    assert restored is not None
    assert restored.status == INTERRUPTED
    assert "daemon stopped" in restored.error


def test_a_queued_job_is_dequeued_rather_than_cancelled(tmp_path: Path) -> None:
    jobs = JobQueue(runner=None, devices=1, directory=tmp_path)
    job = jobs.submit(RunSpec(model="krea2", dataset="poses"))

    assert jobs.cancel(job.id) == q.CANCELLED
    assert job.status == q.CANCELLED
    assert jobs.cancel(job.id) is None, "cancelling twice is not a thing"


def test_a_failing_job_does_not_stop_the_queue(tmp_path: Path) -> None:
    calls = {"n": 0}

    def runner(job, emit, cancel):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("cuda fell over")
        return Result()

    jobs = JobQueue(runner=runner, devices=1, directory=tmp_path)
    first = jobs.submit(RunSpec(model="krea2", dataset="a"))
    assert first.wait(10)
    second = jobs.submit(RunSpec(model="krea2", dataset="b"))
    assert second.wait(10)

    assert first.status == q.FAILED and second.status == q.DONE


def test_run_spec_requires_a_model_and_a_dataset() -> None:
    with pytest.raises(ValueError, match="model"):
        RunSpec.from_dict({"dataset": "poses"})
    with pytest.raises(ValueError, match="dataset"):
        RunSpec.from_dict({"model": "krea2"})


def test_run_spec_keeps_unknown_keys_for_the_backends(tmp_path: Path) -> None:
    spec = RunSpec.from_dict({"model": "krea2", "dataset": "poses", "timestep_type": "shift"})
    assert spec.extra["timestep_type"] == "shift"


# --------------------------------------------------------------------------
# jobs over the API
# --------------------------------------------------------------------------


def test_a_node_reports_whether_its_backend_could_actually_run(api: httpx.Client) -> None:
    """A registered backend with no checkout is worth knowing about early."""
    listing = api.get("/jobs").json()
    assert listing["runner"] is True, "the queue has something to run jobs with"

    backends = api.get("/models").json()["backends"]
    assert backends["aitoolkit"]["ready"] is False, "no aitoolkit_path in the test config"


def test_an_unknown_model_raises_rather_than_falling_through(api: httpx.Client) -> None:
    """v1's detect_backend ends with `return 'kohya'`. This one does not."""
    api.app_state.attach_job_runner(lambda job, emit, cancel: Result())  # type: ignore[attr-defined]

    response = api.post("/jobs", json={"model": "some-unknown-model", "dataset": "poses"})
    assert response.status_code == 422
    assert "no backend handles model" in response.json()["error"]


def test_a_device_this_node_does_not_have_is_refused(api: httpx.Client) -> None:
    response = api.post("/jobs", json={"model": "krea2", "dataset": "poses", "device": 7})
    assert response.status_code == 422


def test_job_endpoints(api: httpx.Client) -> None:
    # Detach the runner so the job stays queued: this is about the endpoints,
    # not about what a job does once it starts.
    api.app_state.jobs.runner = None  # type: ignore[attr-defined]
    job_id = api.post("/jobs", json={"model": "krea2", "dataset": "poses"}).json()["id"]

    assert api.get(f"/jobs/{job_id}").json()["spec"]["model"] == "krea2"
    assert api.get(f"/jobs/{job_id}/loss").json()["points"] == []
    assert api.get("/jobs/nope").status_code == 404

    cancelled = api.delete(f"/jobs/{job_id}").json()
    assert cancelled["dequeued"] is True
    assert api.delete(f"/jobs/{job_id}").status_code == 409

# --------------------------------------------------------------------------
# which backend a job actually runs on
# --------------------------------------------------------------------------


def test_a_job_uses_this_node_s_backend_not_the_shared_registry(state) -> None:
    """The registry is process-global; the node's own backend is not.

    ``core.backends`` keys one dict by backend name, so the last thing to
    call ``register`` owns it for the whole process. An unconfigured
    instance winning that race is silent: it knows every model, so dispatch
    succeeds, and the run then reports `aitoolkit_path is not set` on a node
    whose config.toml has it set. A job resolves through the state, which
    nothing else can overwrite.
    """
    from fluxkrea.core.backends import register, registered
    from fluxkrea.core.backends.aitoolkit import AIToolkitBackend

    configured = state.configured_backends["aitoolkit"]
    assert state.backend_for("flux2-klein-9b") is configured

    # Something else registers a bare instance afterwards - an import, a
    # second State, a test. The registry now points at it.
    intruder = AIToolkitBackend()
    register(intruder)
    try:
        assert registered()["aitoolkit"] is intruder
        assert state.backend_for("flux2-klein-9b") is configured
    finally:
        register(configured)


def test_the_node_report_describes_the_backend_a_job_would_use(state) -> None:
    # "ready: false" while a run uses a different, working instance - or the
    # reverse - is worse than either answer on its own.
    from fluxkrea.core.backends import register
    from fluxkrea.core.backends.aitoolkit import AIToolkitBackend

    configured = state.configured_backends["aitoolkit"]
    register(AIToolkitBackend())
    try:
        reported = state.backends()["aitoolkit"]["ready"]
        assert reported is bool(configured.available())
    finally:
        register(configured)


# --------------------------------------------------------------------------
# fair share
# --------------------------------------------------------------------------


def queued(*projects: str, directory: Path) -> JobQueue:
    """A queue holding one job per argument, submitted in that order."""
    jobs = JobQueue(directory=directory)
    for index, project in enumerate(projects):
        jobs.submit(RunSpec(model="flux2", dataset="d", name=f"run-{index}", project=project))
    return jobs


def test_one_project_alone_still_runs_first_come_first_served(tmp_path: Path) -> None:
    """A single-operator node must behave exactly as FIFO did."""
    jobs = queued("alice", "alice", "alice", directory=tmp_path)
    assert [j.spec.name for j in jobs.waiting()] == ["run-0", "run-1", "run-2"]


def test_a_second_student_is_not_stuck_behind_the_first_ones_batch(tmp_path: Path) -> None:
    """The failure this exists to prevent: five runs at nine, nobody else until lunch."""
    jobs = queued("alice", "alice", "alice", "bob", directory=tmp_path)
    order = [j.project for j in jobs.waiting()]

    assert order[:2] == ["alice", "bob"], f"bob waited behind alice's batch: {order}"
    assert order == ["alice", "bob", "alice", "alice"]


def test_within_a_round_it_is_still_the_order_they_arrived(tmp_path: Path) -> None:
    jobs = queued("alice", "bob", "carol", directory=tmp_path)
    assert [j.project for j in jobs.waiting()] == ["alice", "bob", "carol"]


def test_unclaimed_runs_each_get_their_own_lane(tmp_path: Path) -> None:
    """CLI and fleet runs carry no project; bundling them would make one
    operator's scripted batch the slow lane for every other operator's."""
    jobs = queued("", "", "alice", directory=tmp_path)
    assert [j.project for j in jobs.waiting()] == ["", "", "alice"]


def test_position_counts_the_whole_queue_not_one_project(tmp_path: Path) -> None:
    """The number a student is actually asking for."""
    jobs = queued("alice", "bob", "alice", directory=tmp_path)
    waiting = jobs.waiting()

    assert [jobs.position(j.id) for j in waiting] == [0, 1, 2]
    assert jobs.position("no-such-job") == -1


def test_the_queue_reports_who_is_waiting(api: httpx.Client) -> None:
    """A shared queue is only useful if it says whose runs are in it."""
    # Nothing picks the jobs up, so they stay where this test can read them.
    # The ordering itself is exercised against JobQueue above; what is under
    # test here is the shape of the answer.
    api.app_state.jobs.runner = None  # type: ignore[attr-defined]

    for project in ("alice", "bob"):
        response = api.post(
            "/jobs",
            json={"model": "flux2", "dataset": "d", "project": project, "name": f"{project}-run"},
        )
        assert response.status_code == 202, response.text

    payload = api.get("/jobs").json()
    assert [entry["project"] for entry in payload["queue"]] == ["alice", "bob"]
    assert payload["depth"] == 2

    mine = api.get("/jobs", params={"project": "bob"}).json()
    assert [job["spec"]["name"] for job in mine["jobs"]] == ["bob-run"]
    # Filtered to bob's job, but the position still counts alice's.
    assert mine["jobs"][0]["position"] == 1
    assert len(mine["queue"]) == 2, "the full queue is shown even when jobs are filtered"
