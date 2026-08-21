"""A training run, end to end, against a stand-in trainer.

The one thing the config tests cannot show is whether the whole path works:
spec in, config rendered, process launched, output parsed, events streamed,
cancellation honoured. So this uses a small Python script that prints
ai-toolkit-shaped output and exits - real subprocess, real pipe, real
parsing, no GPU.

That is the only honest way to test this layer. Mocking the subprocess
would test the mock.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from fluxkrea.core.backends import BackendError
from fluxkrea.core.backends.aitoolkit import AIToolkitBackend
from fluxkrea.core.backends.process import ProcessRunner
from fluxkrea.core.backends.spec import RunSpec
from fluxkrea.core.events import Collector, Finished, LossPoint, Progress

#: A stand-in for ai-toolkit's run.py: same output shapes, no torch.
FAKE_TRAINER = '''
import sys, time

config = sys.argv[1] if len(sys.argv) > 1 else ""
print("Running job: fake")
print(f"Loading config {config}")
total = 6
for step in range(1, total + 1):
    print(f"step: {step}/{total} loss: {0.5 / step:.4f}")
    sys.stdout.flush()
    time.sleep(0.05)
print("Saving checkpoint")
'''

FAKE_TRAINER_THAT_FAILS = '''
import sys
print("Loading Flux2 model")
print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
sys.exit(1)
'''

FAKE_TRAINER_THAT_HANGS = '''
import sys, time
print("Running job: fake")
sys.stdout.flush()
for step in range(1, 100000):
    print(f"step: {step}/100000")
    sys.stdout.flush()
    time.sleep(0.05)
'''


def toolkit_with(script: str, tmp_path: Path) -> Path:
    """A folder that looks enough like an ai-toolkit checkout."""
    root = tmp_path / "ai-toolkit"
    root.mkdir(parents=True, exist_ok=True)
    (root / "run.py").write_text(script, encoding="utf-8")
    return root


@pytest.fixture
def dataset_with_masks(dataset: Path) -> Path:
    from fluxkrea.core import paths
    from tests.conftest import make_mask

    for index in range(1, 5):
        make_mask(paths.masks_dir(dataset) / f"punch_{index:03d}.png", size=(64 + index, 48 + index))
    return dataset


# --------------------------------------------------------------------------
# the process runner
# --------------------------------------------------------------------------


def test_a_process_streams_its_output(tmp_path: Path) -> None:
    script = tmp_path / "chatty.py"
    script.write_text("print('one')\nprint('two')\n", encoding="utf-8")
    collector = Collector()

    code = ProcessRunner([sys.executable, str(script)]).run(collector)

    assert code == 0
    assert collector.lines("info") == ["one", "two"]


def test_a_missing_executable_is_reported_not_raised(tmp_path: Path) -> None:
    collector = Collector()
    code = ProcessRunner(["definitely-not-a-real-program-xyz"]).run(collector)

    assert code == -1
    assert any("could not start" in line for line in collector.lines("error"))


def test_arguments_with_spaces_survive(tmp_path: Path) -> None:
    """No shell, so a dataset path with a space in it is not a problem."""
    awkward = tmp_path / "a folder with spaces"
    awkward.mkdir()
    script = tmp_path / "echo.py"
    script.write_text("import sys; print(sys.argv[1])", encoding="utf-8")
    collector = Collector()

    ProcessRunner([sys.executable, str(script), str(awkward)]).run(collector)

    assert collector.lines("info") == [str(awkward)]


def test_a_running_process_can_be_stopped(tmp_path: Path) -> None:
    script = tmp_path / "hang.py"
    script.write_text(FAKE_TRAINER_THAT_HANGS, encoding="utf-8")
    cancel = threading.Event()
    collector = Collector()
    runner = ProcessRunner([sys.executable, str(script)])

    def stop_soon() -> None:
        time.sleep(0.6)
        cancel.set()

    threading.Thread(target=stop_soon, daemon=True).start()
    started = time.monotonic()
    runner.run(collector, cancel)

    assert time.monotonic() - started < 20, "the process was not stopped"
    assert not runner.is_running()


# --------------------------------------------------------------------------
# the backend, driving a real process
# --------------------------------------------------------------------------


def test_a_run_produces_progress_and_loss(tmp_path: Path) -> None:
    backend = AIToolkitBackend(
        toolkit_with(FAKE_TRAINER, tmp_path),
        python_exe=sys.executable,
        output_root=tmp_path / "runs",
    )
    spec = RunSpec(model="flux2", dataset=str(tmp_path), name="probe", steps=6)
    config_path = backend.generate_config(spec)
    collector = Collector()

    backend.start(config_path, collector, total_steps=6)

    steps = [e.step for e in collector.of(Progress)]
    assert steps == [1, 2, 3, 4, 5, 6]
    losses = [e.value for e in collector.of(LossPoint)]
    assert len(losses) == 6
    assert losses[0] > losses[-1], "loss should have been parsed as it fell"
    assert not backend.is_running()


def test_the_generated_config_is_what_gets_launched(tmp_path: Path) -> None:
    """The trainer echoes its argument, so this proves the wiring."""
    backend = AIToolkitBackend(
        toolkit_with("import sys; print('CONFIG=' + sys.argv[1])", tmp_path),
        python_exe=sys.executable,
        output_root=tmp_path / "runs",
        # Krea 2 has no public repo; a run of it names its own checkpoint.
        model_paths={"krea2": "D:/weights/krea2_raw.safetensors"},
    )
    spec = RunSpec(model="krea2", dataset=str(tmp_path), name="probe", steps=1)
    config_path = backend.generate_config(spec)
    collector = Collector()

    backend.start(config_path, collector)

    assert any(str(config_path) in line for line in collector.lines("info"))


def test_a_failing_run_raises_with_the_exit_code(tmp_path: Path) -> None:
    backend = AIToolkitBackend(
        toolkit_with(FAKE_TRAINER_THAT_FAILS, tmp_path),
        python_exe=sys.executable,
        output_root=tmp_path / "runs",
    )
    spec = RunSpec(model="flux2", dataset=str(tmp_path), name="probe", steps=1)
    collector = Collector()

    with pytest.raises(BackendError, match="exited with code 1"):
        backend.start(backend.generate_config(spec), collector)

    assert any("out of memory" in line for line in collector.lines("error"))


def test_a_cancelled_run_does_not_raise(tmp_path: Path) -> None:
    """Stopping on purpose is not a failure."""
    backend = AIToolkitBackend(
        toolkit_with(FAKE_TRAINER_THAT_HANGS, tmp_path),
        python_exe=sys.executable,
        output_root=tmp_path / "runs",
    )
    spec = RunSpec(model="flux2", dataset=str(tmp_path), name="probe", steps=100000)
    config_path = backend.generate_config(spec)
    cancel = threading.Event()
    collector = Collector()

    threading.Timer(0.6, cancel.set).start()
    backend.start(config_path, collector, cancel)  # must not raise

    assert any("stopped" in line.lower() for line in collector.lines("warning"))


def test_a_missing_toolkit_is_a_clear_error(tmp_path: Path) -> None:
    backend = AIToolkitBackend(tmp_path / "not-here", output_root=tmp_path)
    spec = RunSpec(model="flux2", dataset=str(tmp_path), name="probe")

    with pytest.raises(BackendError, match="ai-toolkit not found"):
        backend.start(tmp_path / "nothing.yaml", Collector())


# --------------------------------------------------------------------------
# through the queue, as a job
# --------------------------------------------------------------------------


def test_a_job_runs_through_the_queue(tmp_path: Path, dataset_with_masks: Path) -> None:
    """Spec in, events out, the whole way through the daemon's machinery."""
    from fluxkrea.core.backends import register
    from fluxkrea.core.config import load
    from fluxkrea.daemon.registry import Registry
    from fluxkrea.daemon.runner import make_job_runner
    from fluxkrea.daemon.state import State

    config = load(use_file=False)
    config.dataset.min_resolution = 0
    config.backends.aitoolkit_path = toolkit_with(FAKE_TRAINER, tmp_path)
    config.backends.python_exe = sys.executable
    config.backends.output_root = tmp_path / "runs"

    state = State(config=config, registry=Registry(file=tmp_path / "registry.json"))
    register(AIToolkitBackend.from_config(config))
    state.jobs.runner = make_job_runner(state)
    state.registry.register(dataset_with_masks, "poses")

    job = state.jobs.submit(
        RunSpec(model="flux2", dataset="poses", name="blizzard", steps=6, mask_path="masks")
    )
    assert job.wait(60), "the job never finished"

    try:
        assert job.status == "done", job.error
        assert job.progress == {"step": 6, "total": 6}
        assert job.loss and job.loss[0][1] > job.loss[-1][1]
        assert job.config_path

        import yaml

        rendered = yaml.safe_load(Path(job.config_path).read_text(encoding="utf-8"))
        dataset_block = rendered["config"]["process"][0]["datasets"][0]
        assert dataset_block["mask_path"].endswith("/masks")
        assert dataset_block["folder_path"] == dataset_with_masks.as_posix()

        kinds = [e.event.kind for e in job.events_since(-1)]
        assert kinds.count("finished") == 1
    finally:
        state.shutdown()


def test_a_masked_run_refuses_a_dataset_with_missing_masks(
    tmp_path: Path, dataset: Path
) -> None:
    """ai-toolkit trains an unmasked image silently; catch it before the run."""
    from fluxkrea.core.backends import register
    from fluxkrea.core.config import load
    from fluxkrea.daemon.registry import Registry
    from fluxkrea.daemon.runner import make_job_runner
    from fluxkrea.daemon.state import State

    config = load(use_file=False)
    config.dataset.min_resolution = 0
    config.backends.aitoolkit_path = toolkit_with(FAKE_TRAINER, tmp_path)
    config.backends.python_exe = sys.executable
    config.backends.output_root = tmp_path / "runs"

    state = State(config=config, registry=Registry(file=tmp_path / "registry.json"))
    register(AIToolkitBackend.from_config(config))
    state.jobs.runner = make_job_runner(state)
    state.registry.register(dataset, "poses")  # only one of four has a mask

    job = state.jobs.submit(
        RunSpec(model="flux2", dataset="poses", name="masked", steps=6, mask_path="masks")
    )
    assert job.wait(60)

    try:
        assert job.status == "failed"
        assert "validation errors" in job.error
        lines = [e.event.line for e in job.events_since(-1) if e.event.kind == "log"]
        assert any("missing_mask" in line or "no mask" in line for line in lines)
    finally:
        state.shutdown()
