"""The manifest, the diff, and the tar round trip that sync rests on."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from fluxkrea.core import paths
from fluxkrea.core.dataset import archive, manifest
from tests.conftest import make_image, make_mask


def test_a_manifest_covers_the_bundle_and_nothing_else(dataset: Path) -> None:
    (dataset / "notes.md").write_text("not part of the dataset", encoding="utf-8")

    entries = {e.path for e in manifest.build(dataset)}

    assert "punch_001.jpg" in entries
    assert "punch_001.txt" in entries
    assert "masks/punch_001.png" in entries
    assert "notes.md" not in entries


def test_paths_are_relative_and_posix(dataset: Path) -> None:
    """The same manifest on a Windows laptop and a Linux node, or every file
    reads as changed because one side used backslashes."""
    for entry in manifest.build(dataset):
        assert "\\" not in entry.path
        assert not entry.path.startswith("/")
        assert ":" not in entry.path


def test_digests_can_be_skipped_for_speed(dataset: Path) -> None:
    assert all(e.digest for e in manifest.build(dataset))
    assert not any(e.digest for e in manifest.build(dataset, digests=False))


def test_sidecars_only_is_the_cheap_loop(dataset: Path) -> None:
    """Once images are on a node, a re-masked pass moves kilobytes."""
    full = manifest.build(dataset)
    small = manifest.build(dataset, sidecars_only=True)

    assert small.bytes < full.bytes
    assert all(not e.path.endswith(".jpg") for e in small)
    assert any(e.path.startswith("masks/") for e in small)
    assert any(e.path.endswith(".txt") for e in small)


def test_diff_of_identical_folders_is_empty(dataset: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(dataset, copy)

    diff = manifest.build(dataset).diff(manifest.build(copy))
    assert diff.in_sync
    assert diff.summary() == "in sync"


def test_diff_finds_new_changed_and_extra(dataset: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(dataset, copy)

    make_image(dataset / "punch_005.jpg")
    (dataset / "punch_001.txt").write_text("a different caption entirely", encoding="utf-8")
    (copy / "punch_009.txt").write_text("only on the target", encoding="utf-8")

    diff = manifest.build(dataset).diff(manifest.build(copy))

    assert [e.path for e in diff.added] == ["punch_005.jpg"]
    assert [e.path for e in diff.changed] == ["punch_001.txt"]
    assert [e.path for e in diff.removed] == ["punch_009.txt"]
    assert len(diff.transfers) == 2, "removals are reported, never acted on"


def test_diff_falls_back_to_size_and_mtime_without_digests(dataset: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(dataset, copy)

    assert manifest.build(dataset, digests=False).diff(manifest.build(copy, digests=False)).in_sync


def test_a_same_size_change_is_caught_by_the_digest(dataset: Path, tmp_path: Path) -> None:
    """Which is the reason digests are the default."""
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(dataset, copy)
    original = (dataset / "punch_001.txt").read_text(encoding="utf-8")
    (copy / "punch_001.txt").write_text("x" * len(original), encoding="utf-8")

    diff = manifest.build(dataset).diff(manifest.build(copy))
    assert [e.path for e in diff.changed] == ["punch_001.txt"]


def test_manifest_serialises_and_reloads(dataset: Path) -> None:
    original = manifest.build(dataset)
    reloaded = manifest.Manifest.from_dict(original.as_dict())

    assert len(reloaded) == len(original)
    assert original.diff(reloaded).in_sync


def test_manifest_needs_a_folder(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        manifest.build(tmp_path / "nope")


# --------------------------------------------------------------------------
# tar
# --------------------------------------------------------------------------


def test_tar_round_trip_preserves_the_bundle(dataset: Path, tmp_path: Path) -> None:
    payload = b"".join(archive.stream(dataset))
    target = tmp_path / "restored"

    result = archive.extract(target, io.BytesIO(payload))

    assert result.ok
    assert (target / "punch_001.jpg").read_bytes() == (dataset / "punch_001.jpg").read_bytes()
    assert (target / "punch_001.txt").read_bytes() == (dataset / "punch_001.txt").read_bytes()
    assert (paths.masks_dir(target) / "punch_001.png").is_file()


def test_tar_can_carry_a_subset(dataset: Path, tmp_path: Path) -> None:
    """What a push sends: only the files the diff named."""
    payload = b"".join(archive.stream(dataset, ["punch_002.txt", "masks/punch_001.png"]))
    target = tmp_path / "restored"

    result = archive.extract(target, io.BytesIO(payload))

    assert result.files == 2
    assert (target / "punch_002.txt").is_file()
    assert (paths.masks_dir(target) / "punch_001.png").is_file()
    assert not (target / "punch_001.jpg").exists()


def test_streaming_does_not_hold_the_whole_archive(dataset: Path) -> None:
    make_image(dataset / "big.jpg", size=(1200, 900))
    chunks = list(archive.stream(dataset, chunk=4096))
    assert len(chunks) > 1, "the archive came out in one piece"


def test_extract_reports_what_it_refused(tmp_path: Path) -> None:
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        good = tarfile.TarInfo("punch_001.txt")
        good.size = 3
        tar.addfile(good, io.BytesIO(b"abc"))
        bad = tarfile.TarInfo("../escape.txt")
        bad.size = 3
        tar.addfile(bad, io.BytesIO(b"abc"))
    buffer.seek(0)

    result = archive.extract(tmp_path / "target", buffer)

    assert result.files == 1
    assert not result.ok
    assert "escape.txt" in result.skipped[0]


def test_a_stream_that_is_not_a_tar_is_refused(tmp_path: Path) -> None:
    with pytest.raises(archive.ArchiveError):
        archive.extract(tmp_path, io.BytesIO(b"definitely not a tar"))


def test_masks_survive_a_round_trip_bit_for_bit(dataset_dir: Path, tmp_path: Path) -> None:
    """A mask that changed in transit is a mask that no longer matches its image."""
    make_image(dataset_dir / "a.jpg", size=(64, 48))
    make_mask(paths.masks_dir(dataset_dir) / "a.png", size=(64, 48), box=(5, 5, 10, 10))
    original = (paths.masks_dir(dataset_dir) / "a.png").read_bytes()

    target = tmp_path / "restored"
    archive.extract(target, io.BytesIO(b"".join(archive.stream(dataset_dir))))

    assert (paths.masks_dir(target) / "a.png").read_bytes() == original
