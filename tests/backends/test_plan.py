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
