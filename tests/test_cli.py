"""The CLI, driving a real daemon over HTTP.

Every command here crosses a socket, which is the point: the CLI holds no
dataset logic, so what these tests exercise is the same path a laptop uses
against a lab node. One daemon is shared across the module - starting one
per command would be honest and slow, and the per-command startup is
already covered in ``test_embedded``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from fluxkrea.cli.__main__ import OK, PROBLEM, USAGE, main
from fluxkrea.core import paths
from tests.conftest import make_image


@pytest.fixture(scope="module")
def node(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """One daemon for the module, on a real ephemeral port."""
    from fluxkrea.cli.embedded import EmbeddedDaemon
    from fluxkrea.core.config import load
    from fluxkrea.daemon.registry import Registry
    from fluxkrea.daemon.state import State

    home = tmp_path_factory.mktemp("cli-node")
    config = load(use_file=False)
    config.dataset.min_resolution = 0

    state = State(config=config, registry=Registry(file=home / "registry.json"))
    daemon = EmbeddedDaemon(config, state)
    url = daemon.start()
    yield url
    daemon.stop()
    state.shutdown()


@pytest.fixture(autouse=True)
def use_node(node: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every command at the shared daemon."""
    monkeypatch.setenv("FLUXKREA_URL", node)


def run(*args: str) -> int:
    return main([str(a) for a in args])


def out(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().out


def payload(capsys: pytest.CaptureFixture[str]):  # noqa: ANN201
    return json.loads(out(capsys))


# --------------------------------------------------------------------------
# config and node
# --------------------------------------------------------------------------


def test_config_show(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("--json", "config", "show") == OK
    data = payload(capsys)
    assert data["mask"]["expand"] == 1.6
    assert data["source"] is None


def test_config_init_then_show_reads_it_back(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("config", "init") == OK
    assert paths.config_file().is_file()
    assert run("config", "init") == USAGE, "must not clobber without --force"
    assert run("config", "init", "--force") == OK

    capsys.readouterr()
    assert run("--json", "config", "show") == OK
    assert payload(capsys)["source"] == str(paths.config_file())


def test_config_path_lists_every_location(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("--json", "config", "path") == OK
    located = payload(capsys)
    assert "queue_dir" in located and "fleet_file" in located


def test_a_broken_config_file_is_a_usage_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('[captioner]\napi_key = "sk-nope"\n', encoding="utf-8")
    assert run("--config", bad, "config", "show") == USAGE


def test_node_status(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("--json", "node", "status") == OK
    info = payload(capsys)
    assert "yunet" in info["detectors"]
    assert info["python"].startswith("3.")


def test_node_gpus(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("--json", "node", "gpus") == OK
    assert isinstance(payload(capsys)["gpus"], list)


def test_an_unreachable_node_says_so(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("FLUXKREA_URL", "http://127.0.0.1:1")
    assert run("node", "status") == USAGE
    assert "cannot reach" in capsys.readouterr().err


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------


def test_register_and_list(dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("dataset", "register", dataset) == OK
    capsys.readouterr()

    assert run("--json", "dataset", "list") == OK
    registered = {d["path"] for d in payload(capsys)["datasets"]}
    assert dataset.as_posix() in registered


def test_scan_registers_on_demand(dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Naming a folder is enough; nobody should have to register first."""
    assert run("dataset", "scan", dataset, "--json") == OK
    items = payload(capsys)["items"]
    assert len(items) == 4
    assert items[0]["has_caption"] and items[0]["has_mask"]


def test_scan_of_a_missing_folder_is_a_usage_error(tmp_path: Path) -> None:
    assert run("dataset", "scan", tmp_path / "nope") == USAGE


def test_validate_exit_codes(dataset: Path) -> None:
    assert run("dataset", "validate", dataset) == OK
    assert run("dataset", "validate", dataset, "--require-masks") == PROBLEM


def test_validate_json(dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run("dataset", "validate", dataset, "--require-masks", "--json")
    report = payload(capsys)
    assert report["ok"] is False
    assert report["counts"]["missing_mask"] == 3


def test_resize(dataset_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_image(dataset_dir / "a.jpg", size=(1600, 800))
    assert run("dataset", "resize", dataset_dir, "--size", "800", "--json") == OK

    task = payload(capsys)
    assert task["status"] == "done"
    assert task["result"]["processed"] == 1

    from fluxkrea.core.imaging import read_size

    assert read_size(dataset_dir / "a.jpg").longest == 800


def test_rename_dry_run_moves_nothing(dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("dataset", "rename", dataset, "kick", "--dry-run", "--json") == OK
    task = payload(capsys)

    assert task["result"]["moves"][0]["after"] == "kick_001.jpg"
    assert (dataset / "punch_001.jpg").is_file(), "a dry run must move nothing"


def test_rename(dataset: Path) -> None:
    assert run("dataset", "rename", dataset, "kick") == OK
    assert (dataset / "kick_001.jpg").is_file()
    assert (dataset / "kick_001.txt").is_file()
    assert (paths.masks_dir(dataset) / "kick_001.png").is_file()


def test_rename_reports_a_conflict(dataset: Path) -> None:
    """A bystander sidecar counts too - the whole bundle has to be free."""
    (dataset / "kick_002.txt").write_text("an orphan caption in the way", encoding="utf-8")
    assert run("dataset", "rename", dataset, "kick", "--start", "1", "--dry-run") == PROBLEM


def test_rename_rejects_an_unsafe_prefix(dataset: Path) -> None:
    assert run("dataset", "rename", dataset, "kick/punch") == PROBLEM


def test_augment(dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("dataset", "augment", dataset, "--flip", "--json") == OK
    assert payload(capsys)["result"]["created"] == 4
    assert (dataset / "punch_001_flipHor.jpg").is_file()
    assert (paths.masks_dir(dataset) / "punch_001_flipHor.png").is_file()


def test_augment_with_no_flags_duplicates(dataset: Path) -> None:
    assert run("dataset", "augment", dataset) == OK
    assert (dataset / "punch_001_dup.jpg").is_file()


# --------------------------------------------------------------------------
# masking
# --------------------------------------------------------------------------


def test_detect_finds_nothing_in_flat_test_images(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Real detection, real weights - the fixtures simply have no faces."""
    assert run("dataset", "detect", dataset, "--json") == OK
    result = payload(capsys)["result"]
    assert result["scanned"] == 4
    assert len(result["empty"]) == 4


def test_an_unknown_detector_is_refused(dataset: Path) -> None:
    assert run("dataset", "detect", dataset, "--detector", "insightface") == PROBLEM


def test_boxes_can_be_drawn_by_hand(dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Masking works with no detector at all - the fallback path."""
    assert run("dataset", "boxes", dataset, "punch_001", "--add", "10,10,20,20", "--reviewed",
               "--json") == OK
    stored = payload(capsys)
    assert stored["boxes"][0]["src"] == "manual"
    assert stored["reviewed"] is True


def test_a_malformed_box_is_refused(dataset: Path) -> None:
    assert run("dataset", "boxes", dataset, "punch_001", "--add", "10,10") == USAGE
    assert run("dataset", "boxes", dataset, "punch_001", "--add", "10,10,0,5") == USAGE


def test_boxes_clear(dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run("dataset", "boxes", dataset, "punch_001", "--add", "1,2,3,4")
    capsys.readouterr()
    assert run("dataset", "boxes", dataset, "punch_001", "--clear", "--json") == OK
    assert payload(capsys)["boxes"] == []


def test_review_reports_incomplete_until_every_image_is_seen(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("dataset", "review", dataset, "--json") == PROBLEM
    assert payload(capsys)["reviewed"] == 0

    assert run("dataset", "review", dataset, "--mark-all-reviewed", "--json") == OK
    assert payload(capsys)["complete"] is True


def test_review_can_mark_one_image(dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run("dataset", "review", dataset, "--mark", "punch_001.jpg", "--json")
    assert payload(capsys)["reviewed"] == 1


def test_mask_export_refuses_before_review(dataset: Path) -> None:
    assert run("dataset", "mask", dataset, "--no-detect") == PROBLEM
    assert not (paths.masks_dir(dataset) / "punch_002.png").exists()


def test_the_whole_masking_pipeline_by_hand(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Draw a box, review, export, validate - all over HTTP."""
    run("dataset", "boxes", dataset, "punch_001", "--add", "10,10,20,15", "--reviewed")
    run("dataset", "review", dataset, "--mark-all-reviewed")
    capsys.readouterr()

    assert run("dataset", "mask", dataset, "--no-detect", "--force", "--json") == OK
    result = payload(capsys)["result"]
    assert result["written"] == 4
    assert result["previews"] == 1

    assert run("dataset", "validate", dataset, "--require-masks") == OK


def test_mask_expansion_can_be_changed_without_redetecting(dataset: Path) -> None:
    import numpy as np
    from PIL import Image

    run("dataset", "boxes", dataset, "punch_001", "--add", "10,10,20,15", "--reviewed")
    run("dataset", "review", dataset, "--mark-all-reviewed")

    run("dataset", "mask", dataset, "--no-detect", "--force", "--expand", "1.2", "--feather", "0")
    with Image.open(paths.masks_dir(dataset) / "punch_001.png") as mask:
        tight = int((np.array(mask) == 0).sum())

    run("dataset", "mask", dataset, "--no-detect", "--force", "--expand", "2.5", "--feather", "0")
    with Image.open(paths.masks_dir(dataset) / "punch_001.png") as mask:
        wide = int((np.array(mask) == 0).sum())

    assert wide > tight


# --------------------------------------------------------------------------
# sync
# --------------------------------------------------------------------------


def test_manifest(dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("dataset", "manifest", dataset, "--json") == OK
    entries = {e["path"] for e in payload(capsys)["entries"]}
    assert "masks/punch_001.png" in entries


def test_export_writes_a_tar(dataset: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "poses.tar"
    assert run("dataset", "export", dataset, "--out", out_file) == OK
    assert out_file.stat().st_size > 0

    import tarfile

    with tarfile.open(out_file) as archive:
        assert "punch_001.jpg" in archive.getnames()


def test_push_needs_a_target(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLUXKREA_URL", raising=False)
    assert run("dataset", "push", dataset) == USAGE


def test_push_to_a_url(dataset: Path, tmp_path: Path, node: str, capsys) -> None:
    """The same node stands in for a remote one; only the URL differs."""
    assert run("--url", node, "dataset", "push", dataset, "--dry-run", "--json") == OK
    result = payload(capsys)
    assert result["dry_run"] is True
    assert result["diff"]["in_sync"] is True, "pushing a dataset to itself is a no-op"


# --------------------------------------------------------------------------
# jobs and fleet
# --------------------------------------------------------------------------


def test_jobs_list_reports_the_queue(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("--json", "jobs", "list") == OK
    listing = payload(capsys)
    assert listing["runner"] is True, "the queue has a backend to run jobs with"
    assert listing["devices"] >= 1


def test_node_models_lists_what_can_be_trained(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("--json", "node", "models") == OK
    ids = {m["id"] for m in payload(capsys)["models"]}
    assert {"flux2", "flux2-klein-4b", "krea2"} <= ids


def test_fleet_needs_a_node_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("fleet", "status") == USAGE
    assert "fleet.toml" in capsys.readouterr().err


def test_fleet_status_over_a_configured_node(node: str, capsys) -> None:
    target = paths.fleet_file()
    paths.ensure_dir(target.parent)
    target.write_text(f'[[node]]\nname = "one"\nurl = "{node}"\n', encoding="utf-8")

    assert run("--json", "fleet", "status") == OK
    rows = payload(capsys)
    assert rows[0]["node"] == "one" and rows[0]["state"] == "up"


def test_fleet_nodes_lists_the_configuration(node: str, capsys) -> None:
    target = paths.fleet_file()
    paths.ensure_dir(target.parent)
    target.write_text(f'[[node]]\nname = "one"\nurl = "{node}"\n', encoding="utf-8")

    assert run("--json", "fleet", "nodes") == OK
    assert payload(capsys)[0]["name"] == "one"


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------


def test_json_output_stays_parseable_on_stdout(dataset: Path, capsys) -> None:
    """Human commentary goes to stderr so the JSON can be piped."""
    run("dataset", "boxes", dataset, "punch_001", "--add", "10,10,20,15", "--reviewed")
    run("dataset", "review", dataset, "--mark-all-reviewed")
    capsys.readouterr()

    run("dataset", "mask", dataset, "--no-detect", "--force", "--json")
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "done"


def test_global_flags_work_on_either_side_of_the_subcommand(dataset: Path, capsys) -> None:
    assert run("--json", "dataset", "scan", dataset) == OK
    assert "items" in payload(capsys)
    assert run("dataset", "scan", dataset, "--json") == OK
    assert "items" in payload(capsys)


def test_an_unknown_node_name_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("--node", "does-not-exist", "node", "status") == USAGE
    assert "no node called" in capsys.readouterr().err
