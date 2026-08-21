"""``fk dataset push`` - the manifest diff and the two transports.

The daemon under test stands in for a lab node: a second dataset root, a
second registry, reached over the API exactly as a tunnelled node would be.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from fluxkrea.cli.client import ApiError, Client
from fluxkrea.cli.push import push
from fluxkrea.core import paths
from fluxkrea.core.config import load
from fluxkrea.daemon.app import create_app
from fluxkrea.daemon.registry import Registry
from fluxkrea.daemon.state import State
from tests.conftest import make_image, make_mask


@pytest.fixture
def node(tmp_path: Path) -> Iterator[Client]:
    """A stand-in node with its own root, addressed through the API."""
    root = tmp_path / "node-storage"
    root.mkdir()

    config = load()
    config.dataset.roots = [root]
    config.dataset.min_resolution = 0
    config.daemon.node_name = "olympus-test"

    state = State(config=config, registry=Registry(file=tmp_path / "node-registry.json"))
    http = TestClient(create_app(state=state), base_url="http://olympus-test")
    client = Client(http, base_url="http://olympus-test")
    client.node_root = root  # type: ignore[attr-defined]
    yield client
    state.shutdown()
    http.close()


def test_dry_run_sends_nothing(dataset: Path, node: Client) -> None:
    result = push(node, dataset, dry_run=True)

    assert result.ok
    assert result.files > 0
    assert result.transport == "none"
    assert not list(node.node_root.rglob("*.jpg"))  # type: ignore[attr-defined]


def test_push_creates_the_dataset_on_a_node_that_has_never_seen_it(
    dataset: Path, node: Client
) -> None:
    result = push(node, dataset, transport="tar")

    assert result.ok, result.error
    assert result.dataset_id == "poses"
    target = node.node_root / "poses"  # type: ignore[attr-defined]
    assert (target / "punch_001.jpg").is_file()
    assert (target / "punch_001.txt").is_file()
    assert (paths.masks_dir(target) / "punch_001.png").is_file()


def test_the_copy_is_byte_identical(dataset: Path, node: Client) -> None:
    push(node, dataset, transport="tar")
    target = node.node_root / "poses"  # type: ignore[attr-defined]

    for name in ("punch_001.jpg", "punch_002.txt"):
        assert (target / name).read_bytes() == (dataset / name).read_bytes()
    assert (paths.masks_dir(target) / "punch_001.png").read_bytes() == (
        paths.masks_dir(dataset) / "punch_001.png"
    ).read_bytes()


def test_a_second_push_moves_nothing(dataset: Path, node: Client) -> None:
    push(node, dataset, transport="tar")
    again = push(node, dataset, transport="tar")

    assert again.ok
    assert again.diff.in_sync
    assert again.files == 0


def test_only_the_changed_file_moves(dataset: Path, node: Client) -> None:
    push(node, dataset, transport="tar")
    (dataset / "punch_003.txt").write_text("a completely rewritten caption", encoding="utf-8")

    again = push(node, dataset, transport="tar")

    assert again.files == 1
    target = node.node_root / "poses"  # type: ignore[attr-defined]
    assert "rewritten" in (target / "punch_003.txt").read_text(encoding="utf-8")


def test_sidecars_only_leaves_the_images_alone(dataset: Path, node: Client) -> None:
    """The loop doc 06 exists to make practical: kilobytes, not gigabytes."""
    make_image(dataset / "punch_005.jpg", size=(400, 300))
    full = push(node, dataset, transport="tar")

    make_mask(paths.masks_dir(dataset) / "punch_002.png", size=(66, 50), box=(1, 1, 8, 8))
    (dataset / "punch_001.txt").write_text("a re-captioned frame", encoding="utf-8")

    small = push(node, dataset, transport="tar", sidecars_only=True)

    assert small.ok
    assert small.files == 2
    assert small.bytes < full.bytes / 10
    assert all(not e.path.endswith(".jpg") for e in small.diff.transfers)

    target = node.node_root / "poses"  # type: ignore[attr-defined]
    assert (paths.masks_dir(target) / "punch_002.png").is_file()


def test_the_target_is_rescanned_after_a_push(dataset: Path, node: Client) -> None:
    push(node, dataset, transport="tar")
    items = node.get("/datasets/poses/items")["items"]
    assert len(items) == 4
    assert items[0]["has_mask"]


def test_push_reports_progress(dataset: Path, node: Client, collector) -> None:
    push(node, dataset, transport="tar", emit=collector)

    lines = " ".join(collector.lines())
    assert "Comparing" in lines
    assert "Transferring by tar" in lines


def test_pushing_a_folder_that_is_not_there(node: Client, tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        push(node, tmp_path / "nope")


def test_rsync_without_an_ssh_target_is_refused(dataset: Path, node: Client) -> None:
    with pytest.raises(ValueError, match="rsync needs"):
        push(node, dataset, transport="rsync")


def test_auto_picks_tar_when_rsync_cannot_be_used(dataset: Path, node: Client) -> None:
    """Windows is the reason the fallback has to be good."""
    result = push(node, dataset, transport="auto")
    assert result.transport == "tar"


def test_a_node_with_no_roots_says_where_to_put_it(dataset: Path, tmp_path: Path) -> None:
    config = load()
    config.dataset.roots = []
    state = State(config=config, registry=Registry(file=tmp_path / "empty-registry.json"))
    http = TestClient(create_app(state=state), base_url="http://bare-node")
    client = Client(http, base_url="http://bare-node")

    try:
        with pytest.raises(ApiError, match="Pass --remote-path"):
            push(client, dataset)
    finally:
        state.shutdown()
        http.close()


def test_an_explicit_remote_path_is_honoured(dataset: Path, node: Client) -> None:
    elsewhere = node.node_root / "somewhere-else"  # type: ignore[attr-defined]
    result = push(node, dataset, transport="tar", remote_path=elsewhere.as_posix())

    assert result.ok
    assert (elsewhere / "punch_001.jpg").is_file()


def test_quick_mode_skips_digests(dataset: Path, node: Client) -> None:
    result = push(node, dataset, transport="tar", digests=False, dry_run=True)
    assert result.files > 0
    assert all(not e.digest for e in result.diff.transfers)
