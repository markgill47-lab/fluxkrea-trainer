"""Resize, including the v1 fixes and the mask resampling rule."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PIL import Image

from fluxkrea.core import paths
from fluxkrea.core.dataset import scan
from fluxkrea.core.dataset.ops import resize
from fluxkrea.core.events import Progress
from fluxkrea.core.imaging import read_size
from tests.conftest import make_image, make_mask


def test_resize_fits_the_longest_edge_and_keeps_aspect(dataset_dir: Path) -> None:
    make_image(dataset_dir / "wide.jpg", size=(1600, 800))
    make_image(dataset_dir / "tall.jpg", size=(800, 1600))

    result = resize(dataset_dir, 1024)

    assert result.ok and result.processed == 2
    assert read_size(dataset_dir / "wide.jpg").as_tuple() == (1024, 512)
    assert read_size(dataset_dir / "tall.jpg").as_tuple() == (512, 1024)


def test_correctly_sized_images_are_left_alone(dataset_dir: Path) -> None:
    make_image(dataset_dir / "ok.jpg", size=(1024, 512))
    before = (dataset_dir / "ok.jpg").read_bytes()

    result = resize(dataset_dir, 1024)

    assert result.skipped == 1 and result.processed == 0
    assert (dataset_dir / "ok.jpg").read_bytes() == before, "re-encoded a file it did not need to"


def test_exif_orientation_is_baked_in(dataset_dir: Path) -> None:
    """v1's bug: a 90-degree tag discarded on save, rotating the photo."""
    make_image(dataset_dir / "rotated.jpg", size=(800, 400), exif_orientation=6)

    result = resize(dataset_dir, 800)

    assert result.processed == 1, "a rotated image must be rewritten, not skipped"
    assert read_size(dataset_dir / "rotated.jpg").as_tuple() == (400, 800)
    with Image.open(dataset_dir / "rotated.jpg") as written:
        assert written.getexif().get(0x0112, 1) in (1, None)


def test_file_handles_are_released_so_a_later_rename_works(dataset_dir: Path) -> None:
    """v1's Windows bug: PIL handles left open, blocking the next operation."""
    make_image(dataset_dir / "a.jpg", size=(800, 400))
    resize(dataset_dir, 512)
    (dataset_dir / "a.jpg").rename(dataset_dir / "b.jpg")  # would raise if still open


def test_masks_are_resized_with_the_bundle(dataset_dir: Path) -> None:
    make_image(dataset_dir / "punch_001.jpg", size=(800, 400))
    make_mask(paths.masks_dir(dataset_dir) / "punch_001.png", size=(800, 400), box=(10, 10, 100, 100))

    result = resize(dataset_dir, 400)

    assert result.masks_resized == 1
    assert read_size(paths.masks_dir(dataset_dir) / "punch_001.png").as_tuple() == (400, 200)
    assert read_size(dataset_dir / "punch_001.jpg").as_tuple() == (400, 200)


def test_resized_masks_stay_hard_edged(dataset_dir: Path) -> None:
    """NEAREST, never LANCZOS. A grey pixel is a partial loss weight."""
    make_image(dataset_dir / "punch_001.jpg", size=(800, 400))
    make_mask(paths.masks_dir(dataset_dir) / "punch_001.png", size=(800, 400), box=(11, 13, 101, 97))

    resize(dataset_dir, 401)

    with Image.open(paths.masks_dir(dataset_dir) / "punch_001.png") as mask:
        values = {value for value, count in enumerate(mask.convert("L").histogram()) if count}
    assert values <= {0, 255}, f"resampling introduced partial weights: {sorted(values)}"


def test_mask_and_image_dimensions_stay_equal(dataset_dir: Path) -> None:
    """ai-toolkit warns and swaps sizes when they disagree; never let it."""
    for index, size in enumerate([(801, 399), (640, 641), (1000, 250)], start=1):
        make_image(dataset_dir / f"p_{index}.jpg", size=size)
        make_mask(paths.masks_dir(dataset_dir) / f"p_{index}.png", size=size, box=(1, 1, 5, 5))

    resize(dataset_dir, 512)

    for item in scan(dataset_dir):
        assert read_size(item.image).as_tuple() == read_size(item.mask).as_tuple()


def test_captions_follow_to_an_output_folder(dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "resized"
    result = resize(dataset, 32, output=out)

    assert result.ok
    assert sorted(p.name for p in out.glob("*.txt")) == [
        "punch_001.txt",
        "punch_002.txt",
        "punch_003.txt",
        "punch_004.txt",
    ]
    assert (paths.masks_dir(out) / "punch_001.png").is_file()
    assert (dataset / "punch_001.jpg").is_file(), "the source must be untouched"


def test_untouched_bundles_are_copied_whole_to_an_output_folder(dataset_dir: Path, tmp_path: Path) -> None:
    make_image(dataset_dir / "punch_001.jpg", size=(512, 256))
    (dataset_dir / "punch_001.txt").write_text("caption", encoding="utf-8")
    make_mask(paths.masks_dir(dataset_dir) / "punch_001.png", size=(512, 256))

    out = tmp_path / "out"
    result = resize(dataset_dir, 512, output=out)

    assert result.skipped == 1
    assert (out / "punch_001.jpg").is_file()
    assert (out / "punch_001.txt").is_file()
    assert (paths.masks_dir(out) / "punch_001.png").is_file()


def test_small_images_are_enlarged_to_the_target(dataset_dir: Path) -> None:
    """A bucket of mixed resolutions is worse than a few upscaled images."""
    make_image(dataset_dir / "small.jpg", size=(200, 100))

    result = resize(dataset_dir, 1024)

    assert result.ok and result.processed == 1
    assert not result.too_small
    assert read_size(dataset_dir / "small.jpg").as_tuple() == (1024, 512)


def test_enlarging_is_a_single_resample(dataset_dir: Path) -> None:
    """Two LANCZOS passes to reach one size would soften it for nothing."""
    make_image(dataset_dir / "small.jpg", size=(200, 100))

    resize(dataset_dir, 1024, min_edge=768)

    assert read_size(dataset_dir / "small.jpg").as_tuple() == (1024, 512)


def test_no_upscale_leaves_small_images_alone(dataset_dir: Path) -> None:
    """Still available, for a caller who would rather see the list."""
    make_image(dataset_dir / "small.jpg", size=(200, 100))

    result = resize(dataset_dir, 1024, upscale=False)

    assert result.ok, "declining to enlarge is not an error"
    assert result.too_small and "smaller than the 1024px target" in result.too_small[0][1]
    assert read_size(dataset_dir / "small.jpg").as_tuple() == (200, 100), "left untouched"


def test_a_short_edge_under_the_floor_warns_but_proceeds(dataset_dir: Path, collector) -> None:
    make_image(dataset_dir / "panorama.jpg", size=(4000, 400))

    result = resize(dataset_dir, 1024, min_edge=512, emit=collector)

    assert result.ok and result.processed == 1
    assert any("under the 512px floor" in line for line in collector.lines("warning"))


def test_a_corrupt_file_fails_that_item_only(dataset: Path) -> None:
    (dataset / "broken.jpg").write_bytes(b"not an image")

    result = resize(dataset, 32)

    assert len(result.failed) == 1
    assert result.failed[0][0] == "broken"
    assert result.processed == 4


def test_progress_is_emitted_and_bracketed(dataset: Path, collector) -> None:
    resize(dataset, 32, emit=collector)
    steps = [e.step for e in collector.of(Progress)]
    assert steps == [0, 1, 2, 3, 4]


def test_cancellation_stops_partway(dataset: Path, collector) -> None:
    cancel = threading.Event()

    class CancelAfterOne:
        def __call__(self, event) -> None:
            collector(event)
            if isinstance(event, Progress) and event.step == 2:
                cancel.set()

    result = resize(dataset, 32, emit=CancelAfterOne(), cancel=cancel)
    assert result.total < 4
    assert any("Cancelled" in line for line in collector.lines("warning"))


def test_a_subset_can_be_resized(dataset: Path) -> None:
    items = scan(dataset)
    result = resize(dataset, 32, items=items[:2])

    assert result.total == 2
    assert read_size(dataset / "punch_001.jpg").longest == 32
    assert read_size(dataset / "punch_003.jpg").longest != 32


def test_rejects_a_nonsense_target(dataset: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        resize(dataset, 0)


def test_empty_folder_is_not_an_error(dataset_dir: Path, collector) -> None:
    dataset_dir.mkdir(parents=True)
    result = resize(dataset_dir, 512, emit=collector)
    assert result.ok and result.total == 0
    assert any("No images" in line for line in collector.lines("warning"))
