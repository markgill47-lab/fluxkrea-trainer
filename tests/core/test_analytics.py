"""Loss analytics — the five features v1 gives only to Klein.

Doc 01: Klein alone has trend detection, outlier images, EMA series, metric
export and live config updates, so anything built on the richer API
silently degrades on the other backends. These tests pin the arithmetic
now that it lives above the backend line and every backend gets it.
"""

from __future__ import annotations

import math
import random

import pytest

from fluxkrea.core.analytics import LossSeries
from fluxkrea.core.analytics.loss import _lttb_indices, _quantile, _slope


def falling(count: int = 400, noise: float = 0.0, seed: int = 3) -> LossSeries:
    """A run that improves and flattens, like a real one."""
    rng = random.Random(seed)
    series = LossSeries()
    for step in range(1, count + 1):
        value = 0.5 * math.exp(-step / 100) + 0.04
        if noise:
            value += rng.gauss(0, noise)
        series.add(step, value)
    return series


# --------------------------------------------------------------------------
# EMA
# --------------------------------------------------------------------------


def test_ema_smooths_and_lags() -> None:
    series = falling(300, noise=0.02)
    ema = series.ema(50)

    assert len(ema) == len(series)
    # The EMA of a falling series sits above it, because it remembers.
    assert series.latest_ema(50) > series.latest
    # And it is smoother: less step-to-step movement than the raw signal.
    raw_jitter = sum(abs(b - a) for a, b in zip(series.values, series.values[1:], strict=False))
    ema_jitter = sum(abs(b - a) for a, b in zip(ema, ema[1:], strict=False))
    assert ema_jitter < raw_jitter / 2


def test_ema_starts_at_the_first_value() -> None:
    """No warm-up ramp from zero, which would look like a spike."""
    series = LossSeries()
    series.add(1, 0.5)
    assert series.latest_ema(50) == 0.5


def test_several_windows_are_kept_in_parallel() -> None:
    series = falling(200)
    short = series.latest_ema(10)
    long = series.latest_ema(100)
    assert short is not None and long is not None
    # A shorter window tracks the falling signal more closely.
    assert abs(short - series.latest) < abs(long - series.latest)


def test_ema_is_incremental_not_recomputed() -> None:
    """Appending one point must not depend on the length of the series."""
    a = falling(100)
    b = falling(99)
    b.add(100, 0.5 * math.exp(-100 / 100) + 0.04)
    assert a.latest_ema(50) == pytest.approx(b.latest_ema(50))


# --------------------------------------------------------------------------
# trend
# --------------------------------------------------------------------------


def test_a_falling_run_reads_as_improving() -> None:
    assert falling(300).trend().status == "improving"


def test_a_rising_run_reads_as_degrading() -> None:
    series = LossSeries()
    for step in range(1, 201):
        series.add(step, 0.05 + step * 0.001)
    assert series.trend().status == "degrading"


def test_a_flat_run_converges() -> None:
    """Sustained flatness is convergence, not stalling."""
    series = LossSeries()
    for step in range(1, 400):
        series.add(step, 0.0400)
    assert series.trend().status == "converged"


def test_too_few_points_is_unknown_rather_than_a_guess() -> None:
    series = LossSeries()
    series.add(1, 0.5)
    series.add(2, 0.4)
    trend = series.trend()
    assert trend.status == "unknown"
    assert trend.slope is None


def test_the_slope_is_per_step_not_per_point() -> None:
    """A backend reporting every tenth step is not ten times steeper."""
    dense = LossSeries()
    sparse = LossSeries()
    for index in range(200):
        value = 0.5 - index * 0.001
        dense.add(index + 1, value)
        sparse.add((index + 1) * 10, value)

    dense_slope = dense.trend().slope
    sparse_slope = sparse.trend().slope
    assert dense_slope is not None and sparse_slope is not None
    assert sparse_slope == pytest.approx(dense_slope / 10, rel=0.01)


# --------------------------------------------------------------------------
# outliers
# --------------------------------------------------------------------------


def test_a_consistently_hard_image_is_found() -> None:
    series = LossSeries()
    for step in range(1, 400):
        image = f"img_{step % 20:03d}"
        value = 0.05 + (0.30 if image == "img_007" else 0.0)
        series.add(step, value, image)

    outliers = series.outliers()
    assert [o.image_id for o in outliers] == ["img_007"]
    assert outliers[0].severity > 0
    assert outliers[0].samples > 1


def test_a_uniform_run_has_no_outliers() -> None:
    series = LossSeries()
    for step in range(1, 200):
        series.add(step, 0.05, f"img_{step % 20:03d}")
    assert series.outliers() == []


def test_too_few_images_reports_the_worst_without_claiming_severity() -> None:
    """Under four images there is no meaningful fence, and it says so."""
    series = LossSeries()
    for step, image in enumerate(["a", "b", "a", "b"], start=1):
        series.add(step, 0.9 if image == "a" else 0.1, image)

    outliers = series.outliers()
    assert outliers[0].image_id == "a"
    assert outliers[0].severity == 0.0


def test_points_without_an_image_do_not_become_outliers() -> None:
    series = falling(200)
    assert series.outliers() == []


# --------------------------------------------------------------------------
# robustness
# --------------------------------------------------------------------------


def test_a_nan_loss_is_dropped_rather_than_poisoning_the_series() -> None:
    """A diverged run is real information, but it must not break the rest."""
    series = LossSeries()
    series.add(1, 0.5)
    series.add(2, float("nan"))
    series.add(3, float("inf"))
    series.add(4, 0.4)

    assert len(series) == 2
    assert series.latest == 0.4
    assert math.isfinite(series.latest_ema(50) or 0)


def test_an_empty_series_answers_without_raising() -> None:
    series = LossSeries()
    assert series.latest is None
    assert series.trend().status == "unknown"
    assert series.outliers() == []
    payload = series.as_dict()
    assert payload["points"] == [] and payload["count"] == 0


# --------------------------------------------------------------------------
# decimation
# --------------------------------------------------------------------------


def test_decimation_caps_the_point_count() -> None:
    payload = falling(5000, noise=0.01).as_dict(decimate_to=500)
    assert payload["count"] == 5000
    assert payload["decimated"] is True
    assert len(payload["points"]) <= 500
    assert len(payload["ema"]) == len(payload["points"])


def test_decimation_keeps_the_peak() -> None:
    """A single loss spike is the most interesting thing in the series.

    Stride sampling drops it half the time; LTTB is here to keep it.
    """
    series = LossSeries()
    for step in range(1, 2001):
        series.add(step, 0.05)
    series.add(1500, 0.95)  # the spike

    kept = [point["value"] for point in series.as_dict(decimate_to=200)["points"]]
    assert max(kept) == pytest.approx(0.95)


def test_a_short_series_is_not_decimated() -> None:
    payload = falling(100).as_dict(decimate_to=500)
    assert payload["decimated"] is False
    assert len(payload["points"]) == 100


def test_decimation_preserves_the_endpoints() -> None:
    series = falling(3000)
    payload = series.as_dict(decimate_to=100)
    assert payload["points"][0]["step"] == series.steps[0]
    assert payload["points"][-1]["step"] == series.steps[-1]


def test_lttb_returns_ordered_unique_indices() -> None:
    xs = list(range(500))
    ys = [math.sin(x / 20) for x in xs]
    kept = _lttb_indices(xs, ys, 60)
    assert kept == sorted(kept)
    assert len(set(kept)) == len(kept)
    assert kept[0] == 0 and kept[-1] == 499


# --------------------------------------------------------------------------
# maths
# --------------------------------------------------------------------------


def test_slope_of_a_line() -> None:
    assert _slope([0, 1, 2, 3], [0.0, 2.0, 4.0, 6.0]) == pytest.approx(2.0)


def test_slope_of_a_single_x_is_undefined_not_infinite() -> None:
    assert _slope([5, 5, 5], [1.0, 2.0, 3.0]) is None


def test_quantile_interpolates() -> None:
    ordered = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _quantile(ordered, 0.0) == 0.0
    assert _quantile(ordered, 0.5) == 2.0
    assert _quantile(ordered, 1.0) == 4.0
    assert _quantile(ordered, 0.25) == 1.0


def test_serialisation_carries_everything_the_monitor_draws() -> None:
    series = falling(300, noise=0.01)
    series.add(301, 0.9, "bad_image")
    payload = series.as_dict()

    for key in ("points", "ema", "latest", "latest_ema", "trend", "outliers", "count"):
        assert key in payload
    assert payload["trend"]["status"] in (
        "improving",
        "stable",
        "degrading",
        "converged",
        "unknown",
    )
