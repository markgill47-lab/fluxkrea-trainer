"""How long a run will be, before you start it.

v1's training screen computes ``images x repeats x epochs`` and shows the
total with a time estimate, and it is the most useful thing on that
screen: it is the difference between noticing a two-hour run and
discovering a twenty-hour one at midnight. The arithmetic lives here
rather than in a form, so ``fk train --epochs 6`` gets it too.

**The estimate is measured, not guessed, where it can be.** A seconds-per-
step constant baked into the code is wrong on every card it was not
measured on, and wrong differently at each resolution and rank. This takes
the rate from runs that already finished on this node, and says which it
used. With no history it declines to guess rather than inventing a number
someone will plan an evening around.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

#: Below this many steps a finished run says more about startup - model
#: load, caching, the first sample - than about the per-step rate.
MIN_STEPS_FOR_A_RATE = 50

#: Runs to average over. Enough to shrug off one thermally throttled
#: outlier, few enough to still describe the machine as it is now.
RATE_SAMPLE = 5


@dataclass(frozen=True, slots=True)
class Plan:
    """A run's size, and how long it is likely to take."""

    images: int
    repeats: int
    epochs: int
    steps: int
    #: Measured seconds per step, or None when this node has no history.
    seconds_per_step: float | None
    #: Estimated wall-clock seconds, or None when the rate is unknown.
    seconds: float | None
    #: Where the rate came from, in words - shown next to the estimate so
    #: nobody mistakes an extrapolation for a promise.
    basis: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "repeats": self.repeats,
            "epochs": self.epochs,
            "steps": self.steps,
            "seconds_per_step": self.seconds_per_step,
            "seconds": self.seconds,
            "duration": human_duration(self.seconds),
            "basis": self.basis,
        }


def steps_for(images: int, repeats: int, epochs: int) -> int:
    """``images x repeats x epochs``, the way every trainer counts them.

    One sample per image per repeat, every epoch. Anything below one is
    treated as one: a dataset of zero images is a problem to report, not a
    reason to return a nonsense step count.
    """
    return max(0, int(images)) * max(1, int(repeats)) * max(1, int(epochs))


def plan(
    images: int,
    repeats: int = 1,
    epochs: int = 1,
    *,
    seconds_per_step: float | None = None,
    basis: str = "",
) -> Plan:
    steps = steps_for(images, repeats, epochs)
    seconds = steps * seconds_per_step if seconds_per_step and steps else None
    return Plan(
        images=max(0, int(images)),
        repeats=max(1, int(repeats)),
        epochs=max(1, int(epochs)),
        steps=steps,
        seconds_per_step=seconds_per_step,
        seconds=seconds,
        basis=basis or ("no finished runs to measure against yet"),
    )


def rate_from_history(history: list[dict[str, Any]], model: str = "") -> tuple[float | None, str]:
    """Seconds per step, taken from runs that finished on this node.

    Prefers runs of the same model, because rank and architecture change
    the rate more than anything else does; falls back to any model rather
    than to nothing, and says so. Returns ``(rate, basis)``.
    """
    same = _rates(history, model) if model else []
    if len(same) >= 1:
        return median(same), _describe(len(same), f"{model} run")

    other = _rates(history, "")
    if other:
        return median(other), _describe(len(other), "run", mixed=True)

    return None, "no finished runs to measure against yet"


def _rates(history: list[dict[str, Any]], model: str) -> list[float]:
    found: list[float] = []
    for entry in reversed(history):  # newest first
        if entry.get("status") != "done":
            continue
        if model and str((entry.get("spec") or {}).get("model", "")) != model:
            continue

        started, finished = entry.get("started"), entry.get("finished")
        steps = int((entry.get("progress") or {}).get("step", 0))
        if not started or not finished or steps < MIN_STEPS_FOR_A_RATE:
            continue

        elapsed = float(finished) - float(started)
        if elapsed <= 0:
            continue
        found.append(elapsed / steps)
        if len(found) >= RATE_SAMPLE:
            break
    return found


def _describe(count: int, noun: str, mixed: bool = False) -> str:
    plural = "" if count == 1 else "s"
    where = f"the last {count} {noun}{plural} on this node"
    return f"measured from {where}" + (" (other models)" if mixed else "")


def human_duration(seconds: float | None) -> str:
    """``9h 30m``. Empty string when there is nothing to say."""
    if not seconds or seconds <= 0:
        return ""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    if minutes:
        return f"{minutes}m"
    return f"{total}s"


__all__ = ["MIN_STEPS_FOR_A_RATE", "Plan", "human_duration", "plan", "rate_from_history", "steps_for"]
