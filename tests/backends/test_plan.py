"""Run size and duration - v1's "calculated automatically" box.

The arithmetic is trivial; what these tests pin is the honesty of the
estimate. A seconds-per-step number that came from nowhere is worse than
no number, because someone plans an evening around it.
"""

from __future__ import annotations

from typing import Any

from fluxkrea.core.backends.plan import (
    MIN_STEPS_FOR_A_RATE,
    human_duration,
    plan,
    rate_from_history,
    steps_for,
)


def finished(
    model: str = "krea2",
    steps: int = 1000,
    elapsed: float = 2000.0,
    status: str = "done",
) -> dict[str, Any]:
    return {
        "status": status,
        "started": 1000.0,
        "finished": 1000.0 + elapsed,
        "progress": {"step": steps, "total": steps},
        "spec": {"model": model},
    }


# --------------------------------------------------------------------------
# arithmetic
# --------------------------------------------------------------------------


def test_steps_are_images_times_repeats_times_epochs() -> None:
    """v1's 57 x 10 x 6 = 3420."""
    assert steps_for(57, 10, 6) == 3420


def test_repeats_and_epochs_below_one_are_treated_as_one() -> None:
    assert steps_for(57, 0, 0) == 57
    assert steps_for(57, -3, 1) == 57


def test_an_empty_dataset_is_zero_steps_not_a_nonsense_number() -> None:
    assert steps_for(0, 10, 6) == 0
    assert plan(0, 10, 6).steps == 0


# --------------------------------------------------------------------------
# the estimate
# --------------------------------------------------------------------------


def test_a_known_rate_gives_a_duration() -> None:
    computed = plan(57, 10, 6, seconds_per_step=10.0, basis="measured")
    assert computed.steps == 3420
    assert computed.seconds == 34200
    assert computed.as_dict()["duration"] == "9h 30m"


def test_no_history_declines_to_guess() -> None:
    """Better an honest blank than a number someone plans an evening around."""
    computed = plan(57, 10, 6)
    assert computed.seconds is None
    assert computed.seconds_per_step is None
    assert computed.as_dict()["duration"] == ""
    assert "no finished runs" in computed.basis


def test_the_rate_comes_from_finished_runs_of_the_same_model() -> None:
    rate, basis = rate_from_history([finished("krea2", steps=1000, elapsed=2000)], "krea2")
    assert rate == 2.0
    assert "measured" in basis and "krea2" in basis


def test_a_different_model_is_used_only_as_a_fallback_and_says_so() -> None:
    history = [finished("flux2", steps=1000, elapsed=4000)]
    rate, basis = rate_from_history(history, "krea2")
    assert rate == 4.0
    assert "other models" in basis


def test_the_same_model_wins_over_a_different_one() -> None:
    history = [
        finished("flux2", steps=1000, elapsed=9000),
        finished("krea2", steps=1000, elapsed=2000),
    ]
    rate, _ = rate_from_history(history, "krea2")
    assert rate == 2.0


def test_unfinished_and_failed_runs_are_ignored() -> None:
    history = [
        finished("krea2", status="failed"),
        finished("krea2", status="running"),
        finished("krea2", status="cancelled"),
    ]
    rate, basis = rate_from_history(history, "krea2")
    assert rate is None
    assert "no finished runs" in basis


def test_a_very_short_run_is_startup_not_a_rate() -> None:
    """Model load, caching and the first sample dominate a 20-step run."""
    history = [finished("krea2", steps=MIN_STEPS_FOR_A_RATE - 1, elapsed=600)]
    rate, _ = rate_from_history(history, "krea2")
    assert rate is None


def test_several_runs_are_taken_as_a_median_not_a_mean() -> None:
    """One thermally throttled run must not move the estimate much."""
    history = [
        finished("krea2", steps=1000, elapsed=2000),
        finished("krea2", steps=1000, elapsed=2100),
        finished("krea2", steps=1000, elapsed=90000),  # something went wrong
    ]
    rate, _ = rate_from_history(history, "krea2")
    assert rate == 2.1


def test_a_zero_length_run_does_not_divide_by_it() -> None:
    history = [finished("krea2", steps=1000, elapsed=0.0)]
    rate, _ = rate_from_history(history, "krea2")
    assert rate is None


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------


def test_durations_read_the_way_people_say_them() -> None:
    assert human_duration(34200) == "9h 30m"
    assert human_duration(7200) == "2h"
    assert human_duration(900) == "15m"
    assert human_duration(45) == "45s"


def test_no_duration_is_an_empty_string_not_a_zero() -> None:
    assert human_duration(None) == ""
    assert human_duration(0) == ""


# --------------------------------------------------------------------------
# run naming
# --------------------------------------------------------------------------


def test_a_name_never_contains_a_path() -> None:
    """A 264-character config path, reported as FileNotFoundError.

    The daemon derived the run folder from the dataset's basename and the
    backend slugged the dataset's whole absolute path. They disagreed, the
    config landed outside its own run, and on Windows the doubled path went
    past MAX_PATH - surfacing as a missing file rather than a long name.
    """
    from fluxkrea.core.backends.spec import RunSpec, run_name

    spec = RunSpec(model="flux2", dataset="D:/Projects_26/LoRA_Training_data/Blizzard/Blizzard_Training")
    assert run_name(spec) == "flux2-blizzard-training"
    assert "projects" not in run_name(spec)


def test_the_name_is_the_same_whichever_separator_the_path_used() -> None:
    """The desk is Windows and the fleet is Linux; one run, one name."""
    from fluxkrea.core.backends.spec import RunSpec, run_name

    posix = RunSpec(model="flux2", dataset="D:/data/Blizzard_Training")
    windows = RunSpec(model="flux2", dataset=r"D:\data\Blizzard_Training")
    assert run_name(posix) == run_name(windows) == "flux2-blizzard-training"


def test_a_dataset_id_works_as_well_as_a_path() -> None:
    from fluxkrea.core.backends.spec import RunSpec, run_name

    assert run_name(RunSpec(model="flux2", dataset="blizzard-training")) == "flux2-blizzard-training"


def test_a_trailing_separator_does_not_empty_the_name() -> None:
    from fluxkrea.core.backends.spec import RunSpec, run_name

    assert run_name(RunSpec(model="flux2", dataset="D:/data/poses/")) == "flux2-poses"


def test_an_explicit_name_wins_and_is_slugged() -> None:
    from fluxkrea.core.backends.spec import RunSpec, run_name

    spec = RunSpec(model="flux2", dataset="D:/data/poses", name="Mara v3")
    assert run_name(spec) == "mara-v3"


# --------------------------------------------------------------------------
# where a run writes
# --------------------------------------------------------------------------


def test_ai_toolkit_resolves_our_output_folder_as_its_save_root(tmp_path: Any) -> None:
    """The formula is ai-toolkit's, copied from its source.

        BaseTrainProcess.py:45
        self.save_root = os.path.join(self.training_folder, self.name)

    It appends the job name itself, so handing it the run's own folder put
    checkpoints and samples in `runs/<name>/<name>/` - one level below
    everything that looks for them, including the monitor's sample strip.
    The fake trainer used in tests wrote them where they were expected,
    which is exactly why this needs pinning against the real formula.
    """
    import os
    from pathlib import Path

    from fluxkrea.core.backends.aitoolkit import AIToolkitBackend
    from fluxkrea.core.backends.spec import RunSpec

    backend = AIToolkitBackend(output_root=tmp_path / "runs")
    spec = RunSpec(model="flux2", dataset="D:/data/Blizzard_Training", steps=10)

    config = backend.build(spec)["config"]
    process = config["process"][0]
    save_root = Path(os.path.join(process["training_folder"], config["name"]))

    assert save_root == backend.output_folder(spec)


def test_the_run_folder_is_not_doubled(tmp_path: Any) -> None:
    from fluxkrea.core.backends.aitoolkit import AIToolkitBackend
    from fluxkrea.core.backends.spec import RunSpec

    backend = AIToolkitBackend(output_root=tmp_path / "runs")
    spec = RunSpec(model="flux2", dataset="D:/data/Blizzard_Training", steps=10)

    folder = backend.output_folder(spec)
    assert folder.name != folder.parent.name


def test_an_explicit_output_is_honoured_exactly(tmp_path: Any) -> None:
    """Whatever folder the caller names is the folder the run writes to."""
    import os
    from pathlib import Path

    from fluxkrea.core.backends.aitoolkit import AIToolkitBackend
    from fluxkrea.core.backends.spec import RunSpec

    chosen = tmp_path / "somewhere" / "run-17"
    backend = AIToolkitBackend(output_root=tmp_path / "runs")
    spec = RunSpec(
        model="flux2", dataset="D:/data/poses", output=chosen.as_posix(), steps=10
    )

    config = backend.build(spec)["config"]
    save_root = Path(os.path.join(config["process"][0]["training_folder"], config["name"]))
    assert save_root == chosen
    assert backend.config_path(spec) == chosen / "run-17.yaml"
