"""Box geometry and the review sidecar."""

from __future__ import annotations

from pathlib import Path

import pytest

from fluxkrea.core import paths
from fluxkrea.core.dataset.boxes import BoxStore, ImageBoxes
from fluxkrea.core.detect import MANUAL, Box


def test_expansion_grows_about_the_centre() -> None:
    box = Box(x=100, y=100, w=100, h=100)
    grown = box.expanded(2.0, up_bias=1.0)

    assert grown.x == 50 and grown.w == 200
    assert grown.y == 50 and grown.h == 200


def test_expansion_is_biased_upward_to_catch_the_hairline() -> None:
    """Detectors return eyes-to-chin; hair and hairline carry identity too."""
    box = Box(x=100, y=100, w=100, h=100)
    grown = box.expanded(1.6, up_bias=2.0)

    grew_up = box.y - grown.y
    grew_down = grown.bottom - box.bottom
    assert grew_up > grew_down, "expansion must favour the hairline"


def test_expansion_of_one_changes_nothing() -> None:
    box = Box(x=10, y=20, w=30, h=40)
    assert box.expanded(1.0) == box


def test_expansion_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        Box(0, 0, 10, 10).expanded(0)


def test_clamping_keeps_a_box_inside_the_frame() -> None:
    grown = Box(x=10, y=10, w=100, h=100).expanded(3.0)
    clamped = grown.clamped(120, 90)

    assert clamped.x >= 0 and clamped.y >= 0
    assert clamped.right <= 120 and clamped.bottom <= 90


def test_a_box_entirely_outside_clamps_to_nothing() -> None:
    assert Box(x=500, y=500, w=50, h=50).clamped(100, 100).area == 0


def test_box_round_trips_through_json_shape() -> None:
    box = Box(x=412, y=88, w=96, h=128, src="yunet", conf=0.91)
    assert Box.from_dict(box.as_dict()) == box

    manual = Box.from_dict({"x": 640, "y": 120, "w": 88, "h": 110, "src": MANUAL})
    assert manual.manual and manual.conf is None


def test_a_malformed_box_is_refused() -> None:
    with pytest.raises(ValueError, match="not a box"):
        Box.from_dict({"x": 1, "y": 2})


def test_store_round_trip(tmp_path: Path) -> None:
    store = BoxStore(root=tmp_path)
    store.set_boxes("punch_014.jpg", [Box(1, 2, 3, 4, src="yunet", conf=0.9)], reviewed=True)
    store.save()

    reloaded = BoxStore.load(tmp_path)
    assert reloaded.is_reviewed("punch_014.jpg")
    assert reloaded.boxes("punch_014.jpg")[0].conf == pytest.approx(0.9)


def test_store_accepts_the_bare_doc_shape(tmp_path: Path) -> None:
    paths.boxes_file(tmp_path).write_text(
        '{"punch_014.jpg": {"boxes": [{"x": 1, "y": 2, "w": 3, "h": 4}], "reviewed": true}}',
        encoding="utf-8",
    )
    store = BoxStore.load(tmp_path)
    assert store.is_reviewed("punch_014.jpg")


def test_a_corrupt_box_file_is_loud(tmp_path: Path) -> None:
    """Unlike metadata, this holds human review work. Never silently empty."""
    paths.boxes_file(tmp_path).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not readable as JSON"):
        BoxStore.load(tmp_path)


def test_one_malformed_box_does_not_lose_the_others(tmp_path: Path) -> None:
    paths.boxes_file(tmp_path).write_text(
        '{"a.jpg": {"boxes": [{"x": 1}, {"x": 1, "y": 2, "w": 3, "h": 4}]}}',
        encoding="utf-8",
    )
    assert len(BoxStore.load(tmp_path).boxes("a.jpg")) == 1


def test_redetection_keeps_manual_boxes(tmp_path: Path) -> None:
    """The manual boxes cover the faces the detector could not find."""
    store = BoxStore(root=tmp_path)
    store.set_boxes(
        "a.jpg",
        [Box(1, 1, 10, 10, src="yunet"), Box(50, 50, 20, 20, src=MANUAL)],
        reviewed=True,
    )

    store.record_detection("a.jpg", [Box(2, 2, 12, 12, src="yunet", conf=0.8)])

    kinds = [b.src for b in store.boxes("a.jpg")]
    assert kinds == ["yunet", MANUAL]
    assert not store.is_reviewed("a.jpg"), "new detections need looking at again"


def test_progress_readout(tmp_path: Path) -> None:
    store = BoxStore(root=tmp_path)
    store.set_boxes("a.jpg", [Box(1, 1, 2, 2)], reviewed=True)
    store.set_boxes("b.jpg", [], reviewed=True)
    store.set_boxes("c.jpg", [Box(1, 1, 2, 2)], reviewed=False)

    progress = store.progress(["a.jpg", "b.jpg", "c.jpg", "d.jpg"])

    assert progress.total == 4
    assert progress.reviewed == 2
    assert progress.empty == ["b.jpg", "d.jpg"]
    assert progress.undetected == ["d.jpg"]
    assert "2/4 reviewed" in progress.summary()
    assert not progress.complete


def test_boxes_follow_a_rename(tmp_path: Path) -> None:
    store = BoxStore(root=tmp_path)
    store.set_boxes("punch_001.jpg", [Box(1, 1, 2, 2)], reviewed=True)
    store.apply_rename({"punch_001.jpg": "kick_001.jpg"})
    assert store.is_reviewed("kick_001.jpg")


def test_image_boxes_partition_by_source() -> None:
    entry = ImageBoxes(boxes=[Box(0, 0, 1, 1, src="yunet"), Box(0, 0, 1, 1, src=MANUAL)])
    assert len(entry.detected) == 1
    assert len(entry.manual) == 1
