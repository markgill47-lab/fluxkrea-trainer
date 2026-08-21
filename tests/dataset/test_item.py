"""The bundle invariant."""

from __future__ import annotations

from pathlib import Path

import pytest

from fluxkrea.core import paths
from fluxkrea.core.dataset import DatasetItem


def item_for(root: Path) -> DatasetItem:
    return DatasetItem(
        image=root / "punch_014.jpg",
        caption=root / "punch_014.txt",
        mask=paths.masks_dir(root) / "punch_014.png",
    )


def test_stem_and_root(tmp_path: Path) -> None:
    item = item_for(tmp_path)
    assert item.stem == "punch_014"
    assert item.root == tmp_path
    assert item.suffix == ".jpg"


def test_sidecars_is_the_only_place_that_knows_the_bundle(tmp_path: Path) -> None:
    item = item_for(tmp_path)
    assert list(item.sidecars()) == [item.caption, item.mask]
    assert list(item.members()) == [item.image, item.caption, item.mask]

    bare = DatasetItem(image=tmp_path / "a.jpg")
    assert list(bare.sidecars()) == []
    assert list(bare.members()) == [bare.image]


def test_renamed_to_moves_the_whole_bundle(tmp_path: Path) -> None:
    renamed = item_for(tmp_path).renamed_to("kick_002")

    assert renamed.image == tmp_path / "kick_002.jpg"
    assert renamed.caption == tmp_path / "kick_002.txt"
    assert renamed.mask == paths.masks_dir(tmp_path) / "kick_002.png"


def test_renamed_to_keeps_each_member_where_it_belongs(tmp_path: Path) -> None:
    """The mask stays in masks/ and stays a PNG, or the trainer stops finding it."""
    renamed = item_for(tmp_path).renamed_to("kick_002")
    assert renamed.mask.parent.name == "masks"
    assert renamed.mask.suffix == ".png"
    assert renamed.image.suffix == ".jpg"


def test_renamed_to_survives_a_dotted_filename(tmp_path: Path) -> None:
    item = DatasetItem(image=tmp_path / "punch.014.jpg", caption=tmp_path / "punch.014.txt")
    renamed = item.renamed_to("kick_002")
    assert renamed.image.name == "kick_002.jpg"
    assert renamed.caption.name == "kick_002.txt"


def test_suffixed_is_a_rename(tmp_path: Path) -> None:
    variant = item_for(tmp_path).suffixed("_flipHor")
    assert variant.image.name == "punch_014_flipHor.jpg"
    assert variant.mask.name == "punch_014_flipHor.png"


def test_rebased_puts_the_mask_in_the_new_folders_masks_dir(tmp_path: Path) -> None:
    moved = item_for(tmp_path).rebased(tmp_path / "out")
    assert moved.image == tmp_path / "out" / "punch_014.jpg"
    assert moved.caption == tmp_path / "out" / "punch_014.txt"
    assert moved.mask == paths.masks_dir(tmp_path / "out") / "punch_014.png"


def test_expected_paths_do_not_require_existence(tmp_path: Path) -> None:
    bare = DatasetItem(image=tmp_path / "punch_014.jpg")
    assert bare.expected_caption() == tmp_path / "punch_014.txt"
    assert bare.expected_caption("txt") == tmp_path / "punch_014.txt"
    assert bare.expected_mask() == paths.masks_dir(tmp_path) / "punch_014.png"
    assert bare.expected_preview() == paths.preview_dir(tmp_path) / "punch_014.jpg"


def test_caption_round_trip(tmp_path: Path) -> None:
    item = DatasetItem(image=tmp_path / "punch_014.jpg")
    assert item.read_caption() == ""
    assert not item.has_caption()

    written = item.write_caption("  a fighter mid-punch  ")
    assert written.caption == tmp_path / "punch_014.txt"
    assert written.read_caption() == "a fighter mid-punch"
    assert written.has_caption()


def test_caption_write_returns_an_updated_item(tmp_path: Path) -> None:
    """The caller must keep the result, or its bundle no longer matches disk."""
    item = DatasetItem(image=tmp_path / "punch_014.jpg")
    written = item.write_caption("text")
    assert item.caption is None
    assert written.caption is not None


def test_caption_survives_a_non_utf8_sidecar(tmp_path: Path) -> None:
    (tmp_path / "punch_014.txt").write_bytes(b"caption with a \xff byte")
    item = DatasetItem(image=tmp_path / "punch_014.jpg", caption=tmp_path / "punch_014.txt")
    assert "caption with a" in item.read_caption()


def test_missing_reports_absent_members(tmp_path: Path) -> None:
    item = item_for(tmp_path)
    assert len(item.missing()) == 3
    item.image.write_bytes(b"x")
    assert len(item.missing()) == 2


def test_item_is_immutable(tmp_path: Path) -> None:
    item = item_for(tmp_path)
    with pytest.raises(Exception):
        item.image = tmp_path / "other.jpg"  # type: ignore[misc]


def test_with_helpers_return_new_items(tmp_path: Path) -> None:
    item = item_for(tmp_path)
    assert item.with_quality("good").quality == "good"
    assert item.quality is None
    assert item.with_mask_path(None).mask is None
