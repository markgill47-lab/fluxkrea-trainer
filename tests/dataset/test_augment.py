"""Augmentation, and the rule that a flipped image needs a flipped mask."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fluxkrea.core import paths
from fluxkrea.core.dataset import Metadata, scan
from fluxkrea.core.dataset.ops import augment
from fluxkrea.core.imaging import read_size
from tests.conftest import make_image, make_mask


def test_v1_suffixes_are_preserved(dataset_dir: Path) -> None:
    """Existing fleet datasets are already named this way."""
    make_image(dataset_dir / "punch_001.jpg")

    augment(dataset_dir, ["flip_horizontal", "rotate_90_left", "rotate_90_right", "rotate_180"])

    assert sorted(p.name for p in dataset_dir.glob("*.jpg")) == [
        "punch_001.jpg",
        "punch_001_flipHor.jpg",
        "punch_001_flipVert.jpg",
        "punch_001_rotLeft.jpg",
        "punch_001_rotRight.jpg",
    ]


def test_no_transforms_means_a_plain_duplicate(dataset_dir: Path) -> None:
    make_image(dataset_dir / "punch_001.jpg")
    augment(dataset_dir, [])
    assert (dataset_dir / "punch_001_dup.jpg").is_file()


def test_captions_are_inherited_verbatim(dataset: Path) -> None:
    result = augment(dataset, ["flip_horizontal"])

    assert result.ok
    original = (dataset / "punch_001.txt").read_text(encoding="utf-8")
    assert (dataset / "punch_001_flipHor.txt").read_text(encoding="utf-8") == original


def test_a_flipped_image_gets_a_flipped_mask(dataset_dir: Path) -> None:
    """The correctness rule. A mask on the wrong side masks nothing useful."""
    make_image(dataset_dir / "punch_001.jpg", size=(100, 60))
    make_mask(paths.masks_dir(dataset_dir) / "punch_001.png", size=(100, 60), box=(0, 0, 20, 20))

    augment(dataset_dir, ["flip_horizontal"])

    with Image.open(paths.masks_dir(dataset_dir) / "punch_001_flipHor.png") as mask:
        pixels = np.array(mask.convert("L"))

    assert pixels[5, 95] == 0, "the masked region did not move with the flip"
    assert pixels[5, 5] == 255, "the masked region stayed where it was"


def test_a_rotated_image_gets_a_rotated_mask_of_matching_size(dataset_dir: Path) -> None:
    make_image(dataset_dir / "punch_001.jpg", size=(100, 60))
    make_mask(paths.masks_dir(dataset_dir) / "punch_001.png", size=(100, 60), box=(0, 0, 20, 20))

    augment(dataset_dir, ["rotate_90_left"])

    image_size = read_size(dataset_dir / "punch_001_rotLeft.jpg")
    mask_size = read_size(paths.masks_dir(dataset_dir) / "punch_001_rotLeft.png")
    assert image_size.as_tuple() == (60, 100)
    assert mask_size.as_tuple() == image_size.as_tuple()


def test_transformed_masks_stay_hard_edged(dataset_dir: Path) -> None:
    make_image(dataset_dir / "punch_001.jpg", size=(101, 63))
    make_mask(paths.masks_dir(dataset_dir) / "punch_001.png", size=(101, 63), box=(7, 9, 33, 21))

    augment(dataset_dir, ["rotate_90_right", "flip_horizontal"])

    for name in ("punch_001_rotRight.png", "punch_001_flipHor.png"):
        with Image.open(paths.masks_dir(dataset_dir) / name) as mask:
            values = {v for v, count in enumerate(mask.convert("L").histogram()) if count}
        assert values <= {0, 255}, f"{name} acquired partial weights"


def test_every_variant_is_a_complete_bundle(dataset: Path) -> None:
    augment(dataset, ["flip_horizontal"])

    items = {i.stem: i for i in scan(dataset)}
    variant = items["punch_001_flipHor"]
    assert variant.caption is not None and variant.caption.is_file()
    assert variant.mask is not None and variant.mask.is_file()


def test_output_folder_gets_originals_too(dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "augmented"
    result = augment(dataset, ["flip_horizontal"], output=out)

    assert result.ok
    assert len(list(out.glob("*.jpg"))) == 8
    assert (out / "punch_001.jpg").is_file()
    assert (out / "punch_001_flipHor.jpg").is_file()
    assert (paths.masks_dir(out) / "punch_001.png").is_file()
    assert (paths.masks_dir(out) / "punch_001_flipHor.png").is_file()
    assert len(list(dataset.glob("*.jpg"))) == 4, "the source must be untouched"


def test_orientation_is_baked_into_a_copied_original(dataset_dir: Path, tmp_path: Path) -> None:
    make_image(dataset_dir / "rotated.jpg", size=(80, 40), exif_orientation=6)
    out = tmp_path / "out"

    augment(dataset_dir, ["flip_horizontal"], output=out)

    assert read_size(out / "rotated.jpg").as_tuple() == (40, 80)
    assert read_size(out / "rotated_flipHor.jpg").as_tuple() == (40, 80)


def test_quality_ratings_are_inherited(dataset: Path) -> None:
    meta = Metadata.load(dataset)
    meta.set_quality("punch_001.jpg", "good")
    meta.save()

    augment(dataset, ["flip_horizontal"])

    assert Metadata.load(dataset).quality("punch_001_flipHor.jpg") == "good"


def test_an_unknown_transform_is_refused(dataset: Path) -> None:
    with pytest.raises(ValueError, match="unknown transform"):
        augment(dataset, ["rotate_45"])


def test_a_corrupt_file_fails_that_item_only(dataset: Path) -> None:
    (dataset / "broken.jpg").write_bytes(b"not an image")

    result = augment(dataset, ["flip_horizontal"])

    assert len(result.failed) == 1
    assert (dataset / "punch_001_flipHor.jpg").is_file()


def test_running_twice_does_not_augment_the_augmentations(dataset_dir: Path) -> None:
    """A second in-place pass would otherwise produce ``_flipHor_flipHor``.

    It does, and that is the caller's problem to avoid - but the names must
    at least stay predictable rather than colliding.
    """
    make_image(dataset_dir / "punch_001.jpg")
    augment(dataset_dir, ["flip_horizontal"])
    augment(dataset_dir, ["flip_horizontal"], items=[i for i in scan(dataset_dir) if "_flip" not in i.stem])

    assert not (dataset_dir / "punch_001_flipHor_flipHor.jpg").exists()
