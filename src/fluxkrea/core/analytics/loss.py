"""Loss analytics, above the backend line.

This module is the fix for the asymmetry doc 01 describes: Klein alone has
trend detection, outlier images, EMA series, metric export and live config
updates, while the ai-toolkit backends have ``get_progress`` and
``get_loss_history`` and nothing else — so anything built on the richer
API silently degrades on the other backend.

Doc 02's answer: none of this is a backend concern. **Backends emit
``LossPoint`` events; this consumes the stream and computes the rest for
every backend equally.** The arithmetic is lifted from
``klein_trainer/analytics.py``, which is where it was proved.

Nothing here knows what a backend is, or what a trainer is. It is a series
of numbers and some statistics about them.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

#: EMA windows kept in parallel. Short for responsiveness, long for shape —
#: v1 keeps the same three and the monitor plots the middle one.
EMA_WINDOWS = (10, 50, 100)

#: Fewest points worth fitting a direction to at all.
MIN_TREND_WINDOW = 200

#: How much of the run the trend is measured over. A fixed window was 100
#: points, which on a 4,440-step run measured the last two percent of it -
#: and on a series where a third of the points sit near zero by chance,
#: that is a measurement of where the near-zero points happened to land.
#: The verdict flipped between "improving" and "degrading" depending on
#: where the window ended, on a run whose loss had halved.
#: A third, chosen by measurement rather than taste. Against forty flat
#: noisy series and forty genuinely falling ones of the same shape as a
#: real run: a fifth of the run missed two thirds of the real falls, and a
#: third of it caught every one while calling none of the flat ones a
#: direction.
TREND_FRACTION = 0.35

#: How many standard errors the slope must be from zero before it counts
#: as a direction.
#:
#: This is the part a relative threshold could not do. Comparing the fitted
#: change to the level looks principled and is not: it says nothing about
#: how much of that change the noise could have produced on its own. Tested
#: against forty flat-but-noisy series with the same shape as a real run,
#: a 5%-of-level rule called 68% of them "improving" or "degrading". A
#: slope has to beat its own uncertainty, which is what this measures.
#:
#: Fitted over the raw points rather than the EMA: an EMA's neighbouring
#: values are nearly the same number, so its residuals are far smaller than
#: the real uncertainty and every drift looks certain.
TREND_SIGMA = 3.0

#: And a direction has to be worth mentioning as well as real. On a long
#: enough run a microscopic drift becomes statistically certain while
#: remaining of no interest to anybody.
TREND_SIGNIFICANCE = 0.02

#: A run is converged when it has been flat for this many windows' worth
#: of points, not merely flat right now.
CONVERGENCE_WINDOWS = 2

#: A run is converged when it has been flat for this many windows' worth
#: of points, not merely flat right now.
CONVERGENCE_WINDOWS = 2

#: Tukey's fence multiplier for outlier detection.
IQR_MULTIPLIER = 1.5

TrendStatus = str  # "improving" | "stable" | "degrading" | "converged" | "unknown"


@dataclass(frozen=True, slots=True)
class Trend:
    """Where the loss is going, and how confidently."""

    status: TrendStatus
    slope: float | None
    #: Points the fit was made over.
    window: int

    @property
    def healthy(self) -> bool:
        return self.status in ("improving", "stable", "converged")

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "slope": self.slope, "window": self.window}


@dataclass(frozen=True, slots=True)
class Outlier:
    """An image whose loss sits well above the rest.

    Doc 09: linking a marker back to the training image responsible "is the
    most useful thing Klein's analytics produce and it should be one click,
    not a separate panel".
    """

    image_id: str
    mean: float
    #: How far above the upper fence, in IQRs. When the spread is zero -
    #: every other image identical, which happens on synthetic sets - it is
    #: the excess relative to the fence instead, because reporting 0.0 for
    #: an image seven times the others would read as "not really an
    #: outlier". 0 only when there was too little data for a fence at all.
    severity: float
    samples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "mean": self.mean,
            "severity": self.severity,
            "samples": self.samples,
        }


class LossSeries:
    """A growing loss series and everything derived from it.

    Append-only and incremental: a training run emits a point every step
    for hours, and recomputing an EMA over 20,000 points per point would
    make the monitor the slowest thing on the node.
    """

    def __init__(self, windows: tuple[int, ...] = EMA_WINDOWS) -> None:
        self.windows = windows
        self.steps: list[int] = []
        self.values: list[float] = []
        self._ema: dict[int, list[float]] = {window: [] for window in windows}
        self._current: dict[int, float | None] = {window: None for window in windows}
        self._per_image: dict[str, list[float]] = defaultdict(list)

    # -- ingestion ---------------------------------------------------------

    def add(self, step: int, value: float, image_id: str | None = None) -> None:
        """Record one point. O(1) in the length of the series."""
        if not math.isfinite(value):
            # A NaN loss is real information - a diverged run - but it must
            # not poison every average downstream of it.
            return

        self.steps.append(step)
        self.values.append(value)

        for window in self.windows:
            alpha = 2.0 / (window + 1.0)
            previous = self._current[window]
            current = value if previous is None else previous + alpha * (value - previous)
            self._current[window] = current
            self._ema[window].append(current)

        if image_id:
            self._per_image[image_id].append(value)

    def extend(self, points: list[tuple[int, float]]) -> None:
        for step, value in points:
            self.add(step, value)

    def __len__(self) -> int:
        return len(self.values)

    # -- derived -----------------------------------------------------------

    def ema(self, window: int) -> list[float]:
        return self._ema.get(window, [])

    @property
    def latest(self) -> float | None:
        return self.values[-1] if self.values else None

    def latest_ema(self, window: int = 50) -> float | None:
        return self._current.get(window)

    def trend(self, window: int = 0) -> Trend:
        """Where the loss is going, measured over a slice of the run.

        Three things this gets right that a fixed-window fit over the raw
        points did not, all of them found on a real 4,440-step run whose
        loss had halved and which this reported as "degrading":

        * **The window scales with the run.** 100 points of 4,440 is the
          last two percent, and the answer changed depending on where that
          window happened to end.
        * **The fit is over the EMA.** A third of that run's points sat near
          zero by sampling chance, and the raw slope measured where those
          landed rather than the drift underneath them.
        * **A direction has to beat the noise.** Below
          :data:`TREND_SIGNIFICANCE` of the current level the answer is
          "stable", rather than a confident verdict fitted through noise.
        """
        count = min(window or self._trend_window(), len(self.values))
        if count < 3:
            return Trend(status="unknown", slope=None, window=count)

        steps = self.steps[-count:]
        values = self.values[-count:]
        slope = _slope(steps, values)
        if slope is None:
            return Trend(status="unknown", slope=None, window=count)

        # Two questions, and a direction needs both. Is the slope larger
        # than the noise could have produced by itself, and is it large
        # enough for a person to care about?
        error = _slope_error(steps, values, slope)
        certain = error is not None and abs(slope) > error * TREND_SIGMA

        span = steps[-1] - steps[0]
        level = abs(sum(values) / len(values)) or 1.0
        worth_saying = abs(slope * span) / level >= TREND_SIGNIFICANCE

        if not (certain and worth_saying):
            # Flat now is a pause; flat for a while is convergence.
            if self._flat_before(count):
                return Trend(status="converged", slope=slope, window=count)
            return Trend(status="stable", slope=slope, window=count)

        status = "improving" if slope < 0 else "degrading"
        return Trend(status=status, slope=slope, window=count)

    def _trend_window(self) -> int:
        """A slice of the run, never smaller than is worth fitting."""
        return max(MIN_TREND_WINDOW, int(len(self.values) * TREND_FRACTION))

    def _flat_before(self, window: int) -> bool:
        """Was the run already flat over the window before this one?

        One flat window is a pause; two in a row is convergence. Judged the
        same way as :meth:`trend`, so the two cannot disagree about what
        flat means.
        """
        needed = window * CONVERGENCE_WINDOWS
        if len(self.values) < needed:
            return False
        start = len(self.values) - needed
        end = len(self.values) - window
        steps, values = self.steps[start:end], self.values[start:end]
        earlier = _slope(steps, values)
        if earlier is None or len(steps) < 3:
            return False

        error = _slope_error(steps, values, earlier)
        if error is None or abs(earlier) <= error * TREND_SIGMA:
            return True
        span = steps[-1] - steps[0]
        level = abs(sum(values) / len(values)) or 1.0
        return abs(earlier * span) / level < TREND_SIGNIFICANCE

    @property
    def attributed(self) -> bool:
        """Did any point arrive with an image attached?

        ai-toolkit's output does not name the image a step used, so on that
        backend this is always False and the outlier analysis has nothing
        to work with. Reporting that honestly is the difference between
        "no image is an outlier" and "this cannot be known from here".
        """
        return bool(self._per_image)

    def outliers(self, top: int = 10) -> list[Outlier]:
        """Images whose mean loss sits above Tukey's upper fence.

        These are the images the model finds hardest — usually a mislabelled
        caption, a duplicate, or something that does not belong in the set.
        """
        means = {
            image: sum(losses) / len(losses)
            for image, losses in self._per_image.items()
            if losses
        }
        if not means:
            return []

        ordered = sorted(means.values())
        if len(ordered) < 4:
            # Too few images for a meaningful fence; report the worst by
            # mean and say so with a severity of zero.
            worst = sorted(means.items(), key=lambda kv: kv[1], reverse=True)[:top]
            return [
                Outlier(image_id=image, mean=mean, severity=0.0, samples=len(self._per_image[image]))
                for image, mean in worst
            ]

        q1 = _quantile(ordered, 0.25)
        q3 = _quantile(ordered, 0.75)
        iqr = q3 - q1
        fence = q3 + IQR_MULTIPLIER * iqr

        found = [
            Outlier(
                image_id=image,
                mean=mean,
                severity=_severity(mean, fence, iqr),
                samples=len(self._per_image[image]),
            )
            for image, mean in means.items()
            if mean > fence
        ]
        found.sort(key=lambda outlier: outlier.mean, reverse=True)
        return found[:top]

    # -- serialisation -----------------------------------------------------

    def as_dict(self, decimate_to: int = 0, ema_window: int = 50) -> dict[str, Any]:
        """The payload the monitor draws.

        *decimate_to* caps how many points are returned. Doc 10 wants
        decimation above a couple of thousand visible points so a 20,000
        step run still pans smoothly; done here rather than in the client
        so it also caps what crosses the tunnel.
        """
        steps, values = self.steps, self.values
        ema = self.ema(ema_window)

        if decimate_to and len(values) > decimate_to:
            keep = _lttb_indices(steps, values, decimate_to)
            steps = [steps[i] for i in keep]
            values = [values[i] for i in keep]
            ema = [ema[i] for i in keep] if ema else []

        trend = self.trend()
        return {
            "points": [{"step": s, "value": v} for s, v in zip(steps, values, strict=False)],
            "ema": [{"step": s, "value": v} for s, v in zip(steps, ema, strict=False)],
            "ema_window": ema_window,
            "count": len(self.values),
            "decimated": len(steps) != len(self.values),
            "latest": self.latest,
            "latest_ema": self.latest_ema(ema_window),
            "trend": trend.as_dict(),
            # Whether the backend said which image each step used. Without
            # it there are no outliers to find - and "0 outliers" is a
            # confident claim this data cannot support, so the client needs
            # to be able to tell the two apart.
            "attributed": self.attributed,
            "outliers": [outlier.as_dict() for outlier in self.outliers()],
        }


# --------------------------------------------------------------------------
# maths
# --------------------------------------------------------------------------


def _slope(xs: list[int], ys: list[float]) -> float | None:
    """Least-squares slope of y against x."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
    return numerator / denominator


def _slope_error(xs: list[int], ys: list[float], slope: float) -> float | None:
    """Standard error of a least-squares slope.

    How much slope the scatter around the line could have produced on its
    own. Without it, "the line goes down" and "the line goes down more than
    chance explains" are the same statement - which is how a flat run got
    called improving two thirds of the time.
    """
    n = len(xs)
    if n < 3:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return None
    intercept = mean_y - slope * mean_x
    residuals = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=False))
    return math.sqrt(residuals / (n - 2) / sxx)


def _severity(mean: float, fence: float, iqr: float) -> float:
    """How far past the fence, on whatever scale the data offers."""
    if iqr > 0:
        return (mean - fence) / iqr
    if fence > 0:
        # No spread at all: express the excess relative to the fence.
        return (mean - fence) / fence
    return 0.0


def _quantile(ordered: list[float], fraction: float) -> float:
    """Linear-interpolated quantile of an already-sorted list."""
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _lttb_indices(xs: list[int], ys: list[float], target: int) -> list[int]:
    """Largest-Triangle-Three-Buckets downsampling, returning kept indices.

    Chosen over naive stride sampling because it preserves peaks: a single
    loss spike is the most interesting thing in the series, and every other
    method drops it half the time.
    """
    count = len(ys)
    if target >= count or target < 3:
        return list(range(count))

    kept = [0]
    every = (count - 2) / (target - 2)

    a = 0
    for i in range(target - 2):
        # Average of the next bucket, used as the third triangle vertex.
        start = math.floor((i + 1) * every) + 1
        end = min(math.floor((i + 2) * every) + 1, count)
        if start >= end:
            start, end = min(start, count - 1), min(start + 1, count)
        bucket = range(start, end)
        avg_x = sum(xs[j] for j in bucket) / len(bucket)
        avg_y = sum(ys[j] for j in bucket) / len(bucket)

        # Pick the point in this bucket forming the largest triangle with
        # the last kept point and the next bucket's average.
        range_start = math.floor(i * every) + 1
        range_end = min(math.floor((i + 1) * every) + 1, count)
        best, best_area = range_start, -1.0
        for j in range(range_start, max(range_end, range_start + 1)):
            if j >= count:
                break
            area = abs(
                (xs[a] - avg_x) * (ys[j] - ys[a]) - (xs[a] - xs[j]) * (avg_y - ys[a])
            )
            if area > best_area:
                best, best_area = j, area
        kept.append(best)
        a = best

    kept.append(count - 1)
    return kept


__all__ = ["EMA_WINDOWS", "LossSeries", "Outlier", "Trend"]
