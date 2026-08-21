"""YuNet itself.

Split from ``test_mask.py`` deliberately: everything about the masking
*pipeline* is tested with a fake detector and always runs, while these
tests need the vendored ONNX weights and skip cleanly without them. A
missing 350KB asset must not turn the suite red on a fresh checkout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fluxkrea.core.detect import DetectorError, available, get_detector
from fluxkrea.core.detect.yunet import WEIGHT_NAMES, YuNetDetector, find_weights

needs_weights = pytest.mark.skipif(
    not YuNetDetector.is_available(),
    reason="YuNet weights are not vendored yet; see assets/models/README.md",
)


def test_availability_is_reported_honestly() -> None:
    assert available()["yunet"] == YuNetDetector.is_available()


def test_a_missing_weights_file_says_where_to_put_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FLUXKREA_ASSETS_DIR", str(tmp_path))
    with pytest.raises(DetectorError) as exc:
        YuNetDetector()

    message = str(exc.value)
    assert WEIGHT_NAMES[0] in message
    assert "opencv_zoo" in message, "the error must say where to get them"


def test_an_unknown_detector_name_is_refused() -> None:
    with pytest.raises(DetectorError, match="unknown detector"):
        get_detector("insightface")


def test_the_null_detector_always_works() -> None:
    detector = get_detector("none")
    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


@needs_weights
def test_weights_are_found_where_expected() -> None:
    weights = find_weights()
    assert weights is not None and weights.suffix == ".onnx"


@needs_weights
def test_detects_nothing_in_a_blank_image() -> None:
    detector = YuNetDetector()
    assert detector.detect(np.full((240, 320, 3), 128, dtype=np.uint8)) == []


@needs_weights
def test_rejects_a_non_bgr_array() -> None:
    with pytest.raises(DetectorError, match="3-channel"):
        YuNetDetector().detect(np.zeros((10, 10), dtype=np.uint8))


@needs_weights
def test_boxes_stay_inside_the_frame() -> None:
    """Whatever it finds, a box outside the image would corrupt the mask."""
    detector = YuNetDetector(confidence=0.05)
    image = np.random.default_rng(0).integers(0, 255, (200, 320, 3), dtype=np.uint8)

    for box in detector.detect(image):
        assert box.x >= 0 and box.y >= 0
        assert box.right <= 320 and box.bottom <= 200
        assert box.src == "yunet" and box.conf is not None


@needs_weights
def test_the_same_detector_works_from_several_threads() -> None:
    """``setInputSize`` mutates the engine; each thread must own one."""
    from concurrent.futures import ThreadPoolExecutor

    detector = YuNetDetector()
    sizes = [(120, 200), (200, 120), (300, 300), (90, 400)]
    images = [np.full((h, w, 3), 100, dtype=np.uint8) for h, w in sizes]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(detector.detect, images * 4))

    assert len(results) == 16
