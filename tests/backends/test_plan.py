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
    from fluxkrea.core.backends.aitoolkit import CONFIG_FILENAME

    assert backend.config_path(spec) == chosen / CONFIG_FILENAME


def test_checkpoints_default_to_the_sample_interval() -> None:
    """A checkpoint you cannot see a sample for is hard to judge."""
    from fluxkrea.core.backends.aitoolkit import DEFAULT_SAVE_EVERY, AIToolkitBackend
    from fluxkrea.core.backends.spec import RunSpec

    backend = AIToolkitBackend()

    def save_every(**kwargs: Any) -> int:
        config = backend.build(RunSpec(model="flux2", dataset="d", **kwargs))
        return config["config"]["process"][0]["save"]["save_every"]

    assert save_every(sample_every=400) == 400
    # Explicit wins over both.
    assert save_every(sample_every=400, save_every=800) == 800
    # And with neither, something deliberately not small - checkpoints are
    # hundreds of megabytes each.
    assert save_every() == DEFAULT_SAVE_EVERY


def test_our_config_cannot_be_mistaken_for_a_checkpoint_to_resume_from() -> None:
    """ai-toolkit resumes from anything matching `{job_name}*` in the run
    folder, and calls `torch.load` on it:

        BaseSDTrainProcess.py:816
        patterns = [f"{name}*{post}.safetensors", f"{name}*{post}.pt", f"{name}*{post}"]

    Our config used to be `<run>.yaml` in exactly that folder, so a real
    run announced "RESUMING FROM ... .yaml" and died unpickling it. Run
    names are slugs - [a-z0-9-], no leading punctuation - so a filename
    starting with `_` is outside the pattern for every possible name.
    """
    import fnmatch

    from fluxkrea.core.backends.aitoolkit import CONFIG_FILENAME
    from fluxkrea.core.dataset.naming import slug

    assert CONFIG_FILENAME.startswith("_")
    # And ai-toolkit's own copy of the resolved config is not us.
    assert CONFIG_FILENAME != "config.yaml"

    for raw in ("klein4b smoke", "_fluxkrea", "Mara v3", "fluxkrea", "a"):
        name = slug(raw)
        for pattern in (f"{name}*.safetensors", f"{name}*.pt", f"{name}*"):
            assert not fnmatch.fnmatch(CONFIG_FILENAME, pattern), (name, pattern)


def test_one_loss_point_per_step() -> None:
    """tqdm repaints its bar, sometimes with a new counter and a stale
    postfix. A real 40-step Klein 4B run produced 78 points, each value
    appearing once at step N and again at N+1.
    """
    from fluxkrea.core.backends.aitoolkit import OutputParser
    from fluxkrea.core.events import Collector

    collector = Collector()
    parser = OutputParser(collector, total=20)

    # Exactly the shape the real run produced: the counter moves on before
    # the postfix does.
    for line in (
        " 5%|5    | 1/20 [00:03<01:02, loss: 6.232e-01]",
        " 5%|5    | 1/20 [00:03<01:02, loss: 5.910e-01]",
        "10%|#    | 2/20 [00:06<00:59, loss: 5.910e-01]",
        "10%|#    | 2/20 [00:06<00:59, loss: 5.500e-01]",
    ):
        parser(line)
    parser.flush()

    losses = [e for e in collector.events if e.__class__.__name__ == "LossPoint"]
    # The last value seen while each step was current, once each.
    assert [(e.step, e.value) for e in losses] == [(1, 0.5910), (2, 0.5500)]


def test_the_final_step_is_not_lost_when_the_process_ends() -> None:
    """The last step never advances, so its loss needs an explicit flush."""
    from fluxkrea.core.backends.aitoolkit import OutputParser
    from fluxkrea.core.events import Collector

    collector = Collector()
    parser = OutputParser(collector, total=2)
    parser("| 2/2 [00:06<00:00, loss: 4.200e-01]")
    assert not [e for e in collector.events if e.__class__.__name__ == "LossPoint"]

    parser.flush()
    losses = [e for e in collector.events if e.__class__.__name__ == "LossPoint"]
    assert [(e.step, e.value) for e in losses] == [(2, 0.42)]


def test_the_offload_percentages_are_reachable() -> None:
    """`layer_offloading: true` alone offloads everything - ai-toolkit
    defaults both percentages to 1.0. On a card that is close rather than
    hopeless, a partial offload is the difference between a slow run and
    no run.
    """
    from fluxkrea.core.backends.aitoolkit import AIToolkitBackend
    from fluxkrea.core.backends.spec import RunSpec

    backend = AIToolkitBackend(vram_gb=200.0)
    spec = RunSpec(
        model="flux2",
        dataset="d",
        extra={
            "layer_offloading": True,
            "layer_offloading_transformer_percent": 0.4,
            "layer_offloading_text_encoder_percent": 1.0,
        },
    )
    model_block = backend.build(spec)["config"]["process"][0]["model"]

    assert model_block["layer_offloading"] is True
    assert model_block["layer_offloading_transformer_percent"] == 0.4
    assert model_block["layer_offloading_text_encoder_percent"] == 1.0


def test_the_offload_percentages_are_absent_unless_asked_for() -> None:
    """An unset knob stays unset, so ai-toolkit's own default applies."""
    from fluxkrea.core.backends.aitoolkit import AIToolkitBackend
    from fluxkrea.core.backends.spec import RunSpec

    # A card with room, so the plan does not turn offloading on at all.
    backend = AIToolkitBackend(vram_gb=200.0)
    model_block = backend.build(RunSpec(model="flux2", dataset="d"))["config"]["process"][0]["model"]

    assert model_block["layer_offloading"] is False
    assert "layer_offloading_transformer_percent" not in model_block


# --------------------------------------------------------------------------
# fitting a model on a card
# --------------------------------------------------------------------------


def test_a_model_that_fits_is_left_alone() -> None:
    """The bug, stated as a test: 16.9GB of weights on a 31.8GB card."""
    from fluxkrea.core.backends.memory import plan_memory

    plan = plan_memory(16.9, 31.8)
    assert plan.quantize is False
    assert plan.low_vram is False
    assert plan.layer_offloading is False
    assert "fit" in plan.reason


def test_a_model_that_does_not_fit_is_quantised_before_it_is_offloaded() -> None:
    """Quantising costs precision; offloading costs the PCIe bus."""
    from fluxkrea.core.backends.memory import plan_memory

    plan = plan_memory(24.5, 31.8)
    assert plan.quantize is True
    assert plan.layer_offloading is False


def test_a_model_that_fits_no_way_offloads_only_the_excess() -> None:
    from fluxkrea.core.backends.memory import plan_memory

    plan = plan_memory(64.0, 31.8)
    assert plan.layer_offloading is True
    assert 0.1 <= plan.transformer_percent < 1.0
    assert "slow" in plan.reason


def test_the_same_model_gets_different_answers_on_different_cards() -> None:
    """Which is the whole point - a fleet has both."""
    from fluxkrea.core.backends.memory import plan_memory

    assert plan_memory(16.9, 31.8).quantize is False
    assert plan_memory(16.9, 16.0).quantize is True


def test_low_vram_is_never_chosen_by_the_plan() -> None:
    """It moves the transformer to CPU, which is the thing to avoid.

    Offloading a percentage is the graded version of the same idea; this
    flag is the all-or-nothing one, and the plan never reaches for it.
    """
    from fluxkrea.core.backends.memory import plan_memory

    for weights, vram in ((7.2, 31.8), (24.5, 31.8), (64.0, 31.8), (64.0, 8.0)):
        assert plan_memory(weights, vram).low_vram is False


def test_no_card_is_cautious_rather_than_optimistic() -> None:
    from fluxkrea.core.backends.memory import plan_memory

    plan = plan_memory(16.9, None)
    assert plan.quantize is True
    assert "no GPU" in plan.reason


def test_the_offload_percentages_ride_along_only_when_offloading() -> None:
    from fluxkrea.core.backends.memory import plan_memory

    assert "layer_offloading_transformer_percent" not in plan_memory(7.2, 31.8).as_config()
    assert "layer_offloading_transformer_percent" in plan_memory(64.0, 31.8).as_config()
