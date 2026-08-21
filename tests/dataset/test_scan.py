"""The one scanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from fluxkrea.core import paths
from fluxkrea.core.dataset import Metadata, scan
from fluxkrea.core.dataset.scan import find, natural_key
from tests.conftest import make_image, make_mask


def test_scan_pairs_the_bundle(dataset: Path) -> None:
    items = scan(dataset)
    assert [i.stem for i in items] == ["punch_001", "punch_002", "punch_003", "punch_004"]
    assert all(i.caption is not None for i in items)
    assert items[0].mask == paths.masks_dir(dataset) / "punch_001.png"
    assert items[1].mask is None


def test_scan_skips_its_own_output_folders(dataset: Path) -> None:
    """A recursive scan that picked up masks/ would double the dataset."""
    make_image(paths.preview_dir(dataset) / "punch_001.jpg")
    items = scan(dataset, recursive=True)
    assert [i.stem for i in items] == ["punch_001", "punch_002", "punch_003", "punch_004"]


def test_scan_finds_subfolders_only_when_asked(dataset: Path) -> None:
    make_image(dataset / "extra" / "kick_001.jpg")
    assert len(scan(dataset)) == 4
    assert len(scan(dataset, recursive=True)) == 5


def test_extension_list_comes_from_config_not_the_scanner(dataset: Path) -> None:
    make_image(dataset / "punch_005.webp")
    make_image(dataset / "punch_006.tif")
    assert len(scan(dataset)) == 6
    assert len(scan(dataset, extensions=[".jpg"])) == 4
    assert len(scan(dataset, extensions=["jpg", "webp"])) == 5


def test_extension_matching_is_case_insensitive(dataset_dir: Path) -> None:
    make_image(dataset_dir / "punch_001.JPG")
    assert len(scan(dataset_dir)) == 1


def test_natural_order(dataset_dir: Path) -> None:
    for index in (1, 2, 10, 20, 100):
        make_image(dataset_dir / f"punch_{index}.jpg")
    assert [i.stem for i in scan(dataset_dir)] == [
        "punch_1",
        "punch_2",
        "punch_10",
        "punch_20",
        "punch_100",
    ]


def test_natural_key_is_case_insensitive() -> None:
    assert natural_key("Punch_2") < natural_key("punch_10")


def test_mask_with_the_wrong_extension_is_still_found(dataset_dir: Path) -> None:
    """ai-toolkit matches by basename regardless of extension, so we must too.

    Not seeing it would leave the image apparently unmasked here while the
    trainer happily used it.
    """
    make_image(dataset_dir / "punch_001.jpg")
    make_image(paths.masks_dir(dataset_dir) / "punch_001.jpg")
    item = scan(dataset_dir)[0]
    assert item.mask is not None
    assert item.mask.suffix == ".jpg"


def test_png_mask_wins_when_a_folder_holds_both(dataset_dir: Path) -> None:
    make_image(dataset_dir / "punch_001.jpg")
    make_image(paths.masks_dir(dataset_dir) / "punch_001.jpg")
    make_mask(paths.masks_dir(dataset_dir) / "punch_001.png")
    assert scan(dataset_dir)[0].mask.suffix == ".png"


def test_quality_comes_from_metadata(dataset: Path) -> None:
    meta = Metadata.load(dataset)
    meta.set_quality("punch_002.jpg", "good")
    meta.save()

    items = {i.stem: i for i in scan(dataset)}
    assert items["punch_002"].quality == "good"
    assert items["punch_001"].quality is None


def test_caption_ext_is_configurable(dataset_dir: Path) -> None:
    make_image(dataset_dir / "punch_001.jpg")
    (dataset_dir / "punch_001.caption").write_text("text", encoding="utf-8")
    assert scan(dataset_dir)[0].caption is None
    assert scan(dataset_dir, caption_ext=".caption")[0].caption is not None


def test_scan_rejects_a_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        scan(tmp_path / "nope")


def test_find_one_item(dataset: Path) -> None:
    assert find(dataset, "punch_003") is not None
    assert find(dataset, "nothing") is None


def test_scan_is_cancellable(dataset: Path, collector) -> None:
    import threading

    cancel = threading.Event()
    cancel.set()
    assert scan(dataset, emit=collector, cancel=cancel) == []
