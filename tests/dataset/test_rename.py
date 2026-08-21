"""Rename: the plan/execute split, bundle moves, and the rollback."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from fluxkrea.core import paths
from fluxkrea.core.dataset import Metadata, scan
from fluxkrea.core.dataset.ops import plan_rename, rename
from fluxkrea.core.dataset.ops.rename import TEMP_PREFIX, execute
from tests.conftest import make_image, make_mask


def test_plan_touches_nothing(dataset: Path) -> None:
    before = sorted(p.name for p in dataset.iterdir())
    plan = plan_rename(dataset, "kick")

    assert plan.ok
    assert [m.after.image.name for m in plan.moves] == [
        "kick_001.jpg",
        "kick_002.jpg",
        "kick_003.jpg",
        "kick_004.jpg",
    ]
    assert sorted(p.name for p in dataset.iterdir()) == before, "planning wrote to disk"


def test_rename_moves_the_whole_bundle(dataset: Path) -> None:
    result = rename(dataset, "kick")

    assert result.ok
    assert result.renamed == 4
    assert (dataset / "kick_001.jpg").is_file()
    assert (dataset / "kick_001.txt").is_file()
    assert (paths.masks_dir(dataset) / "kick_001.png").is_file()
    assert not (dataset / "punch_001.jpg").exists()
    assert not (paths.masks_dir(dataset) / "punch_001.png").exists()


def test_the_mask_stays_paired_with_its_image(dataset: Path) -> None:
    """The failure this whole model exists to prevent: a mask left behind.

    ai-toolkit matches masks by basename and silently trains unmasked when
    it finds none, so a desynchronised bundle is invisible until the LoRA
    is wrong.
    """
    original = (paths.masks_dir(dataset) / "punch_001.png").read_bytes()

    rename(dataset, "kick")

    items = {i.stem: i for i in scan(dataset)}
    assert items["kick_001"].mask is not None
    assert items["kick_001"].mask.read_bytes() == original
    assert not any(i.mask for stem, i in items.items() if stem != "kick_001")


def test_captions_keep_their_content(dataset: Path) -> None:
    before = (dataset / "punch_002.txt").read_text(encoding="utf-8")
    rename(dataset, "kick")
    assert (dataset / "kick_002.txt").read_text(encoding="utf-8") == before


def test_renaming_a_folder_onto_its_own_prefix_starts_at_one(dataset: Path) -> None:
    """The two-phase move is what makes a self-collision a non-issue."""
    result = rename(dataset, "punch")

    assert result.ok
    assert result.plan.start_index == 1
    assert sorted(p.name for p in dataset.glob("*.jpg")) == [
        "punch_001.jpg",
        "punch_002.jpg",
        "punch_003.jpg",
        "punch_004.jpg",
    ]


def test_renaming_a_selection_continues_past_bystanders(dataset: Path) -> None:
    """v1's bug: a selection rename restarted at 1 and aborted on collision."""
    items = scan(dataset)
    plan = plan_rename(dataset, "punch", items=items[2:])

    assert plan.start_index == 3
    assert [m.after.image.name for m in plan.moves] == ["punch_003.jpg", "punch_004.jpg"]


def test_explicit_start_index_is_honoured(dataset: Path) -> None:
    plan = plan_rename(dataset, "kick", start_index=100)
    assert [m.after.image.name for m in plan.moves][0] == "kick_100.jpg"


def test_digits_widen_for_large_datasets(dataset_dir: Path) -> None:
    for index in range(1, 12):
        make_image(dataset_dir / f"img_{index}.jpg")
    assert plan_rename(dataset_dir, "k").digits == 3

    for index in range(12, 1200):
        make_image(dataset_dir / f"img_{index}.jpg")
    assert plan_rename(dataset_dir, "k").digits == 4


def test_a_bystander_collision_is_refused_before_anything_moves(dataset: Path) -> None:
    make_image(dataset / "kick_001.jpg")
    punch_001 = [i for i in scan(dataset) if i.stem == "punch_001"]
    plan = plan_rename(dataset, "kick", items=punch_001, start_index=1)

    assert not plan.ok
    assert "kick_001.jpg already exists" in plan.conflicts

    result = execute(plan)
    assert not result.ok
    assert (dataset / "punch_001.jpg").is_file(), "a refused rename must move nothing"


def test_scramble_prefixes_a_letter(dataset: Path) -> None:
    plan = plan_rename(dataset, "kick", scramble=True, seed=7)
    for move in plan.moves:
        stem = move.after.image.stem
        assert stem.startswith("kick_")
        assert stem[5].isalpha() and stem[6:].isdigit()


def test_scramble_is_reproducible_with_a_seed(dataset: Path) -> None:
    first = plan_rename(dataset, "kick", scramble=True, seed=3).describe()
    second = plan_rename(dataset, "kick", scramble=True, seed=3).describe()
    assert first == second


def test_metadata_follows_the_rename(dataset: Path) -> None:
    meta = Metadata.load(dataset)
    meta.set_quality("punch_002.jpg", "good")
    meta.save()

    rename(dataset, "kick")

    assert Metadata.load(dataset).quality("kick_002.jpg") == "good"


def test_rollback_restores_everything_on_failure(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure part way through must leave the folder exactly as it was."""
    before = {p.name: p.read_bytes() for p in dataset.iterdir() if p.is_file()}
    plan = plan_rename(dataset, "kick")

    real_rename = Path.rename
    calls = {"n": 0}

    def flaky(self: Path, target):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 6:
            # Fail once, mid-flight. The rollback that follows must be able
            # to move files back, so only this one call is poisoned.
            raise OSError("disk went away")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky)
    result = execute(plan)

    assert not result.ok and result.rolled_back
    monkeypatch.undo()
    after = {p.name: p.read_bytes() for p in dataset.iterdir() if p.is_file()}
    assert after == before
    assert not list(dataset.glob(f"{TEMP_PREFIX}*"))


def test_cancel_rolls_back_rather_than_leaving_temp_names(dataset: Path) -> None:
    before = sorted(p.name for p in dataset.iterdir())
    cancel = threading.Event()
    cancel.set()

    result = execute(plan_rename(dataset, "kick"), cancel=cancel)

    assert not result.ok and result.rolled_back
    assert sorted(p.name for p in dataset.iterdir()) == before


def test_a_no_op_rename_is_not_an_error(dataset: Path, collector) -> None:
    rename(dataset, "kick")
    result = rename(dataset, "kick", emit=collector)

    assert result.ok and result.renamed == 0
    assert any("already correct" in line for line in collector.lines())


def test_extensions_are_preserved_per_item(dataset_dir: Path) -> None:
    make_image(dataset_dir / "a.jpg")
    make_image(dataset_dir / "b.png")
    make_image(dataset_dir / "c.webp")

    rename(dataset_dir, "shot")

    assert sorted(p.name for p in dataset_dir.iterdir()) == [
        "shot_001.jpg",
        "shot_002.png",
        "shot_003.webp",
    ]


def test_masks_move_within_the_masks_folder(dataset_dir: Path) -> None:
    make_image(dataset_dir / "a.jpg")
    make_mask(paths.masks_dir(dataset_dir) / "a.png")

    rename(dataset_dir, "shot")

    assert (paths.masks_dir(dataset_dir) / "shot_001.png").is_file()
    assert not (dataset_dir / "shot_001.png").exists(), "the mask escaped masks/"


def test_prefix_is_validated(dataset: Path) -> None:
    with pytest.raises(ValueError, match="prefix"):
        plan_rename(dataset, "  ")
    with pytest.raises(ValueError, match="not safe in a filename"):
        plan_rename(dataset, "kick/punch")


def test_plan_serialises_for_the_api(dataset: Path) -> None:
    payload = plan_rename(dataset, "kick").as_dict()
    assert payload["ok"] is True
    assert payload["moves"][0]["before"] == "punch_001.jpg"
    assert payload["moves"][0]["sidecars"] == 2
