"""OpenCV YuNet. The shipping detector (doc 04).

Chosen because it needs no torch and no InsightFace compile pain on Python
3.12, and because ``cv2.FaceDetectorYN`` is already present in every
OpenCV in use here. The weights are a separate ~350KB ONNX file that pip
does not ship; they are vendored into ``assets/models/`` rather than
downloaded at runtime, so the Olympus install script needs no extra
network fetch.

The detector object is **not thread-safe** - ``setInputSize`` mutates it -
so each thread gets its own, held in thread-local storage. Detection runs
threaded over a dataset, and sharing one instance across workers produces
boxes from the wrong image, which is a uniquely difficult bug to see.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from .. import paths
from .base import Box, DetectorError

#: Filenames tried in order, so a newer vendored release is picked up
#: without a code change.
WEIGHT_NAMES = (
    "face_detection_yunet_2023mar.onnx",
    "face_detection_yunet_2022mar.onnx",
    "face_detection_yunet.onnx",
)

WEIGHTS_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def find_weights() -> Path | None:
    """The vendored ONNX file, or ``None`` if it has not been placed yet."""
    for name in WEIGHT_NAMES:
        candidate = paths.model_asset(name)
        if candidate.is_file():
            return candidate
    return None


class YuNetDetector:
    """``cv2.FaceDetectorYN``, wrapped to the :class:`Detector` protocol."""

    name = "yunet"

    def __init__(
        self,
        *,
        confidence: float = 0.5,
        nms: float = 0.3,
        top_k: int = 5000,
        weights: str | Path | None = None,
    ) -> None:
        self.confidence = float(confidence)
        self.nms = float(nms)
        self.top_k = int(top_k)
        self._weights = Path(weights) if weights else find_weights()
        self._local = threading.local()

        if self._weights is None or not self._weights.is_file():
            raise DetectorError(
                "YuNet weights not found. Expected one of "
                f"{', '.join(WEIGHT_NAMES)} in {paths.model_asset(WEIGHT_NAMES[0]).parent}. "
                f"They are ~350KB and vendored into the repo; fetch from {WEIGHTS_URL}"
            )

    @staticmethod
    def is_available() -> bool:
        return hasattr(cv2, "FaceDetectorYN") and find_weights() is not None

    @property
    def weights(self) -> Path:
        assert self._weights is not None  # guarded in __init__
        return self._weights

    def _engine(self):  # noqa: ANN202 - cv2 types are not annotatable
        """One detector per thread. See the module docstring."""
        engine = getattr(self._local, "engine", None)
        if engine is None:
            try:
                engine = cv2.FaceDetectorYN.create(
                    str(self.weights),
                    "",
                    (320, 320),
                    self.confidence,
                    self.nms,
                    self.top_k,
                )
            except cv2.error as exc:
                raise DetectorError(f"could not load YuNet from {self.weights}: {exc}") from exc
            self._local.engine = engine
        return engine

    def detect(self, image: np.ndarray) -> list[Box]:
        """Boxes for one BGR uint8 image.

        Raises rather than returning ``[]`` when detection itself fails: an
        empty list means "no faces here", and a dataset that quietly
        produced no masks is exactly the failure this feature exists to
        prevent.
        """
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise DetectorError("YuNet expects a 3-channel BGR image")

        height, width = image.shape[:2]
        engine = self._engine()

        try:
            engine.setInputSize((width, height))
            _, faces = engine.detect(image)
        except cv2.error as exc:
            raise DetectorError(f"YuNet failed on a {width}x{height} image: {exc}") from exc

        if faces is None:
            return []

        boxes: list[Box] = []
        for face in faces:
            x, y, w, h = (float(v) for v in face[:4])
            score = float(face[-1])
            box = Box(
                x=int(round(x)),
                y=int(round(y)),
                w=int(round(w)),
                h=int(round(h)),
                src=self.name,
                conf=score,
            ).clamped(width, height)
            if box.area > 0:
                boxes.append(box)

        # Largest first: the review UI shows the most significant face first,
        # and a truncated list keeps the ones that matter.
        boxes.sort(key=lambda b: b.area, reverse=True)
        return boxes
