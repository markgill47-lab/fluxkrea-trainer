"""Face masking: the polarity contract, the round trip, and the review gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fluxkrea.core import paths
from fluxkrea.core.dataset import scan, validate
from fluxkrea.core.dataset.boxes import BoxStore
from fluxkrea.core.dataset.ops import (
    detect_faces,
    export_masks,
    render_mask,
    review_order,
    review_progress,
    set_boxes,
)
from fluxkrea.core.dataset.ops.mask import IGNORED, TRAINED
from fluxkrea.core.detect import MANUAL, Box, Detector, DetectorError, NullDetector
from fluxkrea.core.imaging import read_size
from tests.conftest import make_image


class FakeDetector:
    """Returns a fixed box per image. Stands in for YuNet in every test.

    The real detector needs vendored weights; the pipeline around it does
    not, and the pipeline is what this file is about.
    """

    name = "fake"

    def __init__(self, boxes: list[Box] | None = None, fail_on: str | None = None) -> None:
        self.boxes = boxes if boxes is not None else [Box(10, 10, 20, 20, src="fake", conf=0.9)]
        self.fail_on = fail_on
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Box]:
        self.calls += 1
        if self.fail_on and image.shape[1] == int(self.fail_on):
            raise DetectorError("detector exploded")
        return list(self.boxes)


def test_fake_detector_satisfies_the_protocol() -> None:
    assert isinstance(FakeDetector(), Detector)
    assert isinstance(NullDetector(), Detector)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_faces_are_black_and_everything_else_white() -> None:
    """The contract. White is weight 1, i.e. trained."""
    mask = render_mask((100, 80), [Box(40, 30, 20, 20)], expand=1.0, feather=0)
    pixels = np.array(mask)

    assert pixels[0, 0] == TRAINED
    assert pixels[35, 45] == IGNORED
    assert mask.mode == "L"


def test_no_boxes_means_a_fully_trained_mask() -> None:
    pixels = np.array(render_mask((40, 30), [], feather=0))
    assert (pixels == TRAINED).all()


def test_invert_flips_the_polarity() -> None:
    mask = render_mask((50, 50), [Box(10, 10, 10, 10)], expand=1.0, feather=0, invert=True)
    pixels = np.array(mask)
    assert pixels[0, 0] == IGNORED
    assert pixels[15, 15] == TRAINED


def test_an_unfeathered_mask_is_purely_black_and_white() -> None:
    pixels = np.array(render_mask((60, 60), [Box(10, 10, 20, 20)], expand=1.4, feather=0))
    assert set(np.unique(pixels).tolist()) <= {IGNORED, TRAINED}


def test_feathering_is_applied_deliberately_at_generation() -> None:
    """Doc 03: feather here, never acquire it through a resize."""
    pixels = np.array(render_mask((80, 80), [Box(20, 20, 30, 30)], expand=1.0, feather=8))
    values = set(np.unique(pixels).tolist())

    assert values - {IGNORED, TRAINED}, "no gradient at the boundary"
    assert pixels[35, 35] == IGNORED, "the interior must stay fully ignored"
    assert pixels[0, 0] == TRAINED, "distant background must stay fully trained"


def test_expansion_covers_more_than_the_detected_box() -> None:
    tight = np.array(render_mask((100, 100), [Box(40, 40, 20, 20)], expand=1.0, feather=0))
    wide = np.array(render_mask((100, 100), [Box(40, 40, 20, 20)], expand=1.6, feather=0))
    assert (wide == IGNORED).sum() > (tight == IGNORED).sum()


def test_a_box_at_the_edge_does_not_overflow() -> None:
    mask = render_mask((50, 50), [Box(45, 45, 20, 20)], expand=2.0, feather=0)
    assert mask.size == (50, 50)


def test_render_refuses_a_degenerate_size() -> None:
    with pytest.raises(ValueError, match="cannot render"):
        render_mask((0, 10), [])


# --------------------------------------------------------------------------
# detection pass
# --------------------------------------------------------------------------


def test_detection_persists_boxes_and_is_re_runnable(dataset: Path) -> None:
    detector = FakeDetector()
    result = detect_faces(dataset, detector=detector, workers=1)

    assert result.ok
    assert result.scanned == 4 and result.with_faces == 4 and result.boxes == 4
    assert paths.boxes_file(dataset).is_file()

    store = BoxStore.load(dataset)
    assert len(store.boxes("punch_001.jpg")) == 1


def test_detection_flags_images_with_nothing_found(dataset: Path, collector) -> None:
    result = detect_faces(dataset, detector=NullDetector(), workers=1, emit=collector)

    assert len(result.empty) == 4
    assert any("where misses hide" in line for line in collector.lines("warning"))


def test_detection_is_threaded_and_still_records_every_image(dataset: Path) -> None:
    detector = FakeDetector()
    result = detect_faces(dataset, detector=detector, workers=4)

    assert detector.calls == 4
    assert len(BoxStore.load(dataset)) == 4
    assert result.boxes == 4


def test_only_missing_skips_images_already_detected(dataset: Path) -> None:
    detector = FakeDetector()
    detect_faces(dataset, detector=detector, workers=1)
    make_image(dataset / "punch_005.jpg")

    detector.calls = 0
    detect_faces(dataset, detector=detector, workers=1, only_missing=True)

    assert detector.calls == 1


def test_a_detector_failure_is_reported_not_swallowed(dataset: Path) -> None:
    """An empty list means 'no faces here'; a failure must not look like one."""
    make_image(dataset / "wide.jpg", size=(500, 100))
    result = detect_faces(dataset, detector=FakeDetector(fail_on="500"), workers=1)

    assert not result.ok
    assert result.failed[0][0] == "wide"


def test_redetecting_keeps_hand_drawn_boxes(dataset: Path) -> None:
    detect_faces(dataset, detector=FakeDetector(), workers=1)
    set_boxes(
        dataset,
        "punch_001.jpg",
        [*BoxStore.load(dataset).boxes("punch_001.jpg"), Box(30, 30, 8, 8, src=MANUAL)],
    )

    detect_faces(dataset, detector=FakeDetector(), workers=1)

    sources = [b.src for b in BoxStore.load(dataset).boxes("punch_001.jpg")]
    assert MANUAL in sources


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


def reviewed_dataset(root: Path, boxes: list[Box] | None = None) -> BoxStore:
    detect_faces(root, detector=FakeDetector(boxes), workers=1)
    store = BoxStore.load(root)
    for name in list(store):
        store.mark_reviewed(name)
    store.save()
    return store


def test_export_writes_a_mask_per_image_at_native_size(dataset: Path) -> None:
    reviewed_dataset(dataset)
    result = export_masks(dataset)

    assert result.ok and result.written == 4
    for item in scan(dataset):
        assert item.mask is not None, f"{item.stem} has no mask"
        assert read_size(item.mask).as_tuple() == read_size(item.image).as_tuple()


def test_export_uses_the_layout_the_trainer_reads(dataset: Path) -> None:
    """``punch_014.jpg`` -> ``masks/punch_014.png``, matched by basename."""
    reviewed_dataset(dataset)
    export_masks(dataset)

    assert (paths.masks_dir(dataset) / "punch_001.png").is_file()
    assert sorted(p.suffix for p in paths.masks_dir(dataset).iterdir()) == [".png"] * 4


def test_the_round_trip_holds(dataset: Path) -> None:
    """Doc 02's mask round-trip test: detect, write, read back, check polarity,
    size and alignment against the source image."""
    box = Box(12, 8, 20, 16, src="fake", conf=0.95)
    reviewed_dataset(dataset, [box])
    export_masks(dataset, expand=1.0, feather=0)

    item = scan(dataset)[0]
    with Image.open(item.mask) as written:
        pixels = np.array(written.convert("L"))

    assert pixels.shape == (read_size(item.image).height, read_size(item.image).width)
    assert pixels[box.y + 2, box.x + 2] == IGNORED, "the face region is not excluded"
    assert pixels[0, 0] == TRAINED, "the background is not trained"


def test_export_refuses_unreviewed_images(dataset: Path, collector) -> None:
    """detect -> review -> export. The review pass is not optional polish."""
    detect_faces(dataset, detector=FakeDetector(), workers=1)
    before = {p.name: p.read_bytes() for p in paths.masks_dir(dataset).glob("*.png")}

    result = export_masks(dataset, emit=collector)

    assert not result.ok
    assert len(result.refused) == 4
    after = {p.name: p.read_bytes() for p in paths.masks_dir(dataset).glob("*.png")}
    assert after == before, "a refused export must write nothing"
    assert any("Refused" in line for line in collector.lines("error"))


def test_export_refuses_zero_detection_images(dataset: Path) -> None:
    detect_faces(dataset, detector=NullDetector(), workers=1)
    store = BoxStore.load(dataset)
    for name in list(store):
        store.mark_reviewed(name)
    store.save()

    result = export_masks(dataset)
    assert not result.ok and len(result.refused) == 4


def test_force_accepts_frames_that_genuinely_contain_no_face(dataset: Path) -> None:
    detect_faces(dataset, detector=NullDetector(), workers=1)

    result = export_masks(dataset, force=True)

    assert result.ok and result.written == 4
    assert len(result.unmasked) == 4
    pixels = np.array(Image.open(paths.masks_dir(dataset) / "punch_001.png").convert("L"))
    assert (pixels == TRAINED).all(), "an unmasked image must train in full"


def test_require_review_can_be_turned_off_in_config(dataset: Path) -> None:
    detect_faces(dataset, detector=FakeDetector(), workers=1)
    assert export_masks(dataset, require_review=False).ok


def test_previews_are_written_as_a_review_aid(dataset: Path) -> None:
    reviewed_dataset(dataset)
    result = export_masks(dataset)

    assert result.previews == 4
    assert (paths.preview_dir(dataset) / "punch_001.jpg").is_file()
    assert read_size(paths.preview_dir(dataset) / "punch_001.jpg").as_tuple() == read_size(
        dataset / "punch_001.jpg"
    ).as_tuple()


def test_previews_can_be_skipped(dataset: Path) -> None:
    reviewed_dataset(dataset)
    result = export_masks(dataset, write_previews=False)
    assert result.previews == 0
    assert not paths.preview_dir(dataset).exists()


def test_masks_are_not_scanned_as_training_data(dataset: Path) -> None:
    reviewed_dataset(dataset)
    export_masks(dataset)
    assert len(scan(dataset)) == 4
    assert len(scan(dataset, recursive=True)) == 4


def test_re_export_at_a_new_expansion_needs_no_detection(dataset: Path) -> None:
    """Changing the expansion factor re-renders from stored boxes (doc 04)."""
    detector = FakeDetector()
    reviewed_dataset(dataset, detector.boxes)
    export_masks(dataset, expand=1.2, feather=0)
    tight = np.array(Image.open(paths.masks_dir(dataset) / "punch_001.png")).copy()

    calls_before = detector.calls
    export_masks(dataset, expand=2.0, feather=0)
    wide = np.array(Image.open(paths.masks_dir(dataset) / "punch_001.png"))

    assert (wide == IGNORED).sum() > (tight == IGNORED).sum()
    assert detector.calls == calls_before, "re-export must not re-detect"


def test_exported_dataset_passes_validation(dataset: Path) -> None:
    reviewed_dataset(dataset)
    export_masks(dataset)

    report = validate(dataset, min_resolution=0, require_masks=True)
    assert report.ok, [str(p) for p in report.problems]


# --------------------------------------------------------------------------
# review support
# --------------------------------------------------------------------------


def test_review_progress_reads_out(dataset: Path) -> None:
    detect_faces(dataset, detector=FakeDetector(), workers=1)
    set_boxes(dataset, "punch_001.jpg", [Box(1, 1, 5, 5, src=MANUAL)], reviewed=True)

    progress = review_progress(dataset)

    assert progress.total == 4 and progress.reviewed == 1
    assert "1/4 reviewed" in progress.summary()


def test_zero_detection_images_sort_to_the_front(dataset: Path) -> None:
    store = BoxStore(root=dataset)
    store.set_boxes("punch_001.jpg", [Box(1, 1, 5, 5)], reviewed=True)
    store.set_boxes("punch_002.jpg", [Box(1, 1, 5, 5)], reviewed=False)
    store.set_boxes("punch_003.jpg", [], reviewed=True)
    store.save()

    order = [i.stem for i in review_order(scan(dataset), store)]

    assert order[0] in ("punch_003", "punch_004"), "empties must come first"
    assert order[-1] == "punch_001", "already reviewed goes last"


def test_set_boxes_marks_reviewed_and_persists(dataset: Path) -> None:
    set_boxes(dataset, "punch_001.jpg", [Box(1, 2, 3, 4, src=MANUAL)])

    store = BoxStore.load(dataset)
    assert store.is_reviewed("punch_001.jpg")
    assert store.boxes("punch_001.jpg")[0].src == MANUAL
