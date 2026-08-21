"""The detector interface, and the box type everything downstream speaks.

Detection is deliberately pluggable (doc 04). YuNet ships, but recall is
the only metric that matters here - a false positive costs a wasted
region, a false negative puts an unmasked face into training and defeats
the entire feature. Martial arts and dance are the hard case: heads turned
away, extreme tilt, motion blur, occlusion by a limb. The escalation path
is a YOLO *head* detector, because head detection beats face detection
when the subject is turned away, and that is the common case.

So the protocol is small enough that swapping the implementation is a
config change, and the boxes it returns are the only thing masking knows
about.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np

#: Where a box came from. ``manual`` boxes are drawn by a human in review
#: and must survive a re-detection pass untouched.
MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned region, in pixels, on the oriented image.

    "Oriented" matters: coordinates are relative to the image *after* EXIF
    rotation is applied, which is what every other part of the pipeline
    sees and what the mask is written against.
    """

    x: int
    y: int
    w: int
    h: int
    src: str = "unknown"
    conf: float | None = None

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    @property
    def manual(self) -> bool:
        return self.src == MANUAL

    def expanded(self, factor: float, up_bias: float = 1.0) -> Box:
        """Grow the box about its centre, biased upward.

        Detectors return an eyes-to-chin box, but hair, hairline and jaw
        carry identity too (doc 04). *up_bias* multiplies the growth above
        the box only, so the extra coverage lands on the hairline rather
        than on the chest.
        """
        if factor <= 0:
            raise ValueError(f"expansion factor must be positive, got {factor}")

        grow_x = (self.w * factor - self.w) / 2
        grow_y = (self.h * factor - self.h) / 2
        top = grow_y * up_bias

        x = round(self.x - grow_x)
        y = round(self.y - top)
        w = round(self.w + grow_x * 2)
        h = round(self.h + top + grow_y)
        return replace(self, x=x, y=y, w=w, h=h)

    def clamped(self, width: int, height: int) -> Box:
        """Clip to the image. An expanded box routinely leaves the frame."""
        x = max(0, min(self.x, width))
        y = max(0, min(self.y, height))
        right = max(0, min(self.right, width))
        bottom = max(0, min(self.bottom, height))
        return replace(self, x=x, y=y, w=max(0, right - x), h=max(0, bottom - y))

    def intersects(self, other: Box) -> bool:
        return not (
            self.right <= other.x
            or other.right <= self.x
            or self.bottom <= other.y
            or other.bottom <= self.y
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "src": self.src}
        if self.conf is not None:
            payload["conf"] = round(float(self.conf), 4)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Box:
        try:
            return cls(
                x=int(round(float(data["x"]))),
                y=int(round(float(data["y"]))),
                w=int(round(float(data["w"]))),
                h=int(round(float(data["h"]))),
                src=str(data.get("src", "unknown")),
                conf=None if data.get("conf") is None else float(data["conf"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a box: {data!r}") from exc


@runtime_checkable
class Detector(Protocol):
    """What masking needs from a detector, and nothing more."""

    name: str

    def detect(self, image: np.ndarray) -> list[Box]:
        """Find faces in a BGR uint8 array. Returns boxes in image pixels."""
        ...


class DetectorError(Exception):
    """A detector that cannot run. Raised loudly rather than returning nothing.

    Returning an empty list on failure would look exactly like "no faces
    here", and silently produce an unmasked training set.
    """


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def get_detector(name: str, **options: Any) -> Detector:
    """Build a detector by name. The config's ``mask.detector`` comes here."""
    key = name.strip().lower()
    if key == "yunet":
        from .yunet import YuNetDetector

        return YuNetDetector(**options)
    if key == "none":
        return NullDetector()
    raise DetectorError(f"unknown detector {name!r}; available: yunet, none")


def available() -> dict[str, bool]:
    """Which detectors could actually run right now, for ``GET /node``."""
    from .yunet import YuNetDetector

    return {"yunet": YuNetDetector.is_available(), "none": True}


class NullDetector:
    """Finds nothing. For datasets masked entirely by hand, and for tests."""

    name = "none"

    def detect(self, image: np.ndarray) -> list[Box]:
        return []
