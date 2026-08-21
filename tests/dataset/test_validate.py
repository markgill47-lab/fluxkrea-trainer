"""Validation - the cheap pass that catches every v1 dataset bug."""

from __future__ import annotations

from pathlib import Path

from fluxkrea.core import paths
from fluxkrea.core.dataset import validate
from tests.conftest import make_image, make_mask


def test_a_clean_dataset_reports_nothing_serious(dataset: Path) -> None:
    report = validate(dataset, min_resolution=0)
    assert report.ok
    assert report.items == 4
    assert not report.problems


def test_missing_and_empty_captions(dataset_dir: Path) -> None:
    make_image(dataset_dir / "a.jpg")
    make_image(dataset_dir / "b.jpg")
    (dataset_dir / "b.txt").write_text("   \n", encoding="utf-8")

    report = validate(dataset_dir, min_resolution=0)

    assert [p.kind for p in report.of_kind("missing_caption")] == ["missing_caption"]
    assert [p.stem for p in report.of_kind("empty_caption")] == ["b"]


def test_orphan_caption_from_a_half_finished_rename(dataset: Path) -> None:
    (dataset / "punch_099.txt").write_text("a caption with no image", encoding="utf-8")

    report = validate(dataset, min_resolution=0)

    assert [p.stem for p in report.of_kind("orphan_caption")] == ["punch_099"]


def test_orphan_mask(dataset: Path) -> None:
    make_mask(paths.masks_dir(dataset) / "punch_099.png")

    report = validate(dataset, min_resolution=0)

    assert [p.stem for p in report.of_kind("orphan_mask")] == ["punch_099"]


def test_missing_masks_only_matter_when_masking_is_enabled(dataset: Path) -> None:
    assert not validate(dataset, min_resolution=0).of_kind("missing_mask")

    report = validate(dataset, min_resolution=0, require_masks=True)
    assert len(report.of_kind("missing_mask")) == 3
    assert not report.ok, "a missing mask trains the face it was meant to exclude"


def test_mask_size_mismatch_is_an_error(dataset_dir: Path) -> None:
    """ai-toolkit warns and tries to swap the dimensions. Catch it here."""
    make_image(dataset_dir / "a.jpg", size=(100, 60))
    make_mask(paths.masks_dir(dataset_dir) / "a.png", size=(60, 100))

    report = validate(dataset_dir, min_resolution=0)

    problems = report.of_kind("mask_size_mismatch")
    assert len(problems) == 1
    assert "60x100" in problems[0].message and "100x60" in problems[0].message
    assert not report.ok


def test_mask_size_accounts_for_exif_orientation(dataset_dir: Path) -> None:
    """A rotated image is 60x100 to every consumer, so its mask should be too."""
    make_image(dataset_dir / "a.jpg", size=(100, 60), exif_orientation=6)
    make_mask(paths.masks_dir(dataset_dir) / "a.png", size=(60, 100))

    assert not validate(dataset_dir, min_resolution=0).of_kind("mask_size_mismatch")


def test_a_jpeg_mask_is_an_error(dataset_dir: Path) -> None:
    make_image(dataset_dir / "a.jpg", size=(64, 48))
    make_image(paths.masks_dir(dataset_dir) / "a.jpg", size=(64, 48))

    report = validate(dataset_dir, min_resolution=0)

    assert report.of_kind("mask_wrong_format")
    assert not report.ok


def test_below_the_resolution_floor_is_a_warning(dataset: Path) -> None:
    report = validate(dataset, min_resolution=512)

    assert len(report.of_kind("below_resolution_floor")) == 4
    assert report.ok, "small images are a warning, not a refusal"


def test_a_corrupt_image_is_an_error(dataset: Path) -> None:
    (dataset / "broken.jpg").write_bytes(b"not an image")

    report = validate(dataset, min_resolution=0)

    assert [p.stem for p in report.of_kind("unreadable_image")] == ["broken"]
    assert not report.ok


def test_duplicate_stems_cannot_have_distinct_sidecars(dataset_dir: Path) -> None:
    make_image(dataset_dir / "punch_001.jpg")
    make_image(dataset_dir / "punch_001.png")

    report = validate(dataset_dir, min_resolution=0)

    assert report.of_kind("duplicate_stem")
    assert not report.ok


def test_case_only_collisions_are_flagged_for_the_fleet(dataset_dir: Path) -> None:
    """Windows tolerates these; the Linux nodes they are rsynced to do not."""
    make_image(dataset_dir / "Punch_001.jpg")
    make_image(dataset_dir / "punch_002.jpg")
    (dataset_dir / "punch_001.png").write_bytes(b"")

    make_image(dataset_dir / "punch_001.webp")
    report = validate(dataset_dir, min_resolution=0)

    assert report.of_kind("case_collision")


def test_grey_in_a_mask_is_reported_as_information(dataset_dir: Path) -> None:
    import numpy as np
    from PIL import Image

    make_image(dataset_dir / "a.jpg", size=(64, 48))
    array = np.full((48, 64), 255, dtype=np.uint8)
    array[10:20, 10:20] = 128
    paths.ensure_dir(paths.masks_dir(dataset_dir))
    Image.fromarray(array, mode="L").save(paths.masks_dir(dataset_dir) / "a.png")

    report = validate(dataset_dir, min_resolution=0)

    assert report.of_kind("mask_has_grey")
    assert report.ok, "feathering is deliberate; this is information, not a failure"


def test_report_serialises_for_the_api(dataset: Path) -> None:
    payload = validate(dataset, min_resolution=512).as_dict()

    assert payload["items"] == 4
    assert payload["ok"] is True
    assert payload["counts"]["below_resolution_floor"] == 4
    assert payload["problems"][0]["severity"] == "warning"


def test_validate_never_modifies(dataset: Path) -> None:
    before = {p.name: p.stat().st_mtime_ns for p in dataset.rglob("*") if p.is_file()}
    validate(dataset, require_masks=True, min_resolution=1024)
    after = {p.name: p.stat().st_mtime_ns for p in dataset.rglob("*") if p.is_file()}
    assert before == after


def test_validate_can_run_on_a_provided_item_list(dataset: Path) -> None:
    from fluxkrea.core.dataset import scan

    items = scan(dataset)[:2]
    assert validate(dataset, items=items, min_resolution=0).items == 2
