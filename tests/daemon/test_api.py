"""API contract tests - doc 06's table, endpoint by endpoint."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from fluxkrea.core import paths
from tests.conftest import make_image
from tests.daemon.conftest import register, run_op


# --------------------------------------------------------------------------
# node
# --------------------------------------------------------------------------


def test_health(api: httpx.Client) -> None:
    payload = api.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["node"] == "test-node"
    assert payload["queue_depth"] == 0


def test_node_reports_the_versions_that_matter(api: httpx.Client) -> None:
    """Blackwell needs torch 2.6+ / CUDA 12.6+; skew is invisible otherwise."""
    payload = api.get("/node").json()
    for key in ("python", "opencv", "torch", "cuda", "driver", "detectors", "gpus"):
        assert key in payload, key


def test_gpus(api: httpx.Client) -> None:
    payload = api.get("/gpus").json()
    assert isinstance(payload["gpus"], list)


def test_config_endpoint_never_leaks_a_secret(
    api: httpx.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLUXKREA_CLAUDE_API_KEY", "sk-not-in-the-response")
    body = api.get("/config").text
    assert "sk-not-in-the-response" not in body


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def test_register_and_list(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    assert dataset_id == "poses"

    listing = api.get("/datasets").json()["datasets"]
    assert listing[0]["id"] == "poses"
    assert listing[0]["exists"] is True


def test_registering_twice_is_idempotent(api: httpx.Client, dataset: Path) -> None:
    first = register(api, dataset)
    second = register(api, dataset)
    assert first == second
    assert len(api.get("/datasets").json()["datasets"]) == 1


def test_two_folders_with_one_name_get_distinct_ids(api: httpx.Client, tmp_path: Path) -> None:
    (tmp_path / "a" / "poses").mkdir(parents=True)
    (tmp_path / "b" / "poses").mkdir(parents=True)
    assert register(api, tmp_path / "a" / "poses") != register(api, tmp_path / "b" / "poses")


def test_a_path_outside_the_roots_is_refused(api: httpx.Client, tmp_path: Path) -> None:
    """The API is remote code execution scoped to the node; scope it."""
    outside = tmp_path.parent / "elsewhere"
    outside.mkdir(exist_ok=True)
    response = api.post("/datasets", json={"path": outside.as_posix()})
    assert response.status_code == 403
    assert "outside the configured dataset roots" in response.json()["error"]


def test_registering_something_that_is_not_a_folder(api: httpx.Client, tmp_path: Path) -> None:
    assert api.post("/datasets", json={"path": (tmp_path / "nope").as_posix()}).status_code == 400
    assert api.post("/datasets", json={}).status_code == 400


def test_create_makes_the_folder_for_a_first_push(api: httpx.Client, tmp_path: Path) -> None:
    target = tmp_path / "brand-new"
    response = api.post("/datasets", json={"path": target.as_posix(), "create": True})
    assert response.status_code == 201
    assert target.is_dir()


def test_unknown_dataset_is_404(api: httpx.Client) -> None:
    assert api.get("/datasets/nope/items").status_code == 404


def test_forget_leaves_the_folder_alone(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    assert api.delete(f"/datasets/{dataset_id}").status_code == 200
    assert dataset.is_dir()
    assert api.get("/datasets").json()["datasets"] == []


# --------------------------------------------------------------------------
# items
# --------------------------------------------------------------------------


def test_items_carry_the_bundle_and_the_review_state(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    payload = api.get(f"/datasets/{dataset_id}/items").json()

    assert len(payload["items"]) == 4
    first = payload["items"][0]
    assert first["stem"] == "punch_001"
    assert first["has_caption"] and first["has_mask"]
    assert payload["review"]["total"] == 4


def test_item_bytes(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    image = api.get(f"/datasets/{dataset_id}/items/punch_001/image")
    assert image.status_code == 200 and image.content[:2] == b"\xff\xd8"

    mask = api.get(f"/datasets/{dataset_id}/items/punch_001/mask")
    assert mask.status_code == 200 and mask.content[:4] == b"\x89PNG"

    assert api.get(f"/datasets/{dataset_id}/items/punch_002/mask").status_code == 404
    assert api.get(f"/datasets/{dataset_id}/items/nothing/image").status_code == 404


def test_caption_round_trip(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    api.put(f"/datasets/{dataset_id}/items/punch_001/caption", json={"caption": "a new caption"})

    assert api.get(f"/datasets/{dataset_id}/items/punch_001/caption").json()["caption"] == "a new caption"
    assert (dataset / "punch_001.txt").read_text(encoding="utf-8").strip() == "a new caption"


def test_validate_endpoint(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)

    assert api.get(f"/datasets/{dataset_id}/validate").json()["ok"] is True
    strict = api.get(f"/datasets/{dataset_id}/validate", params={"require_masks": True}).json()
    assert strict["ok"] is False
    assert strict["counts"]["missing_mask"] == 3


def test_scan_returns_a_count_and_a_report(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    payload = api.post(f"/datasets/{dataset_id}/scan").json()
    assert payload["items"] == 4
    assert payload["validation"]["ok"] is True


# --------------------------------------------------------------------------
# the remote review pass
# --------------------------------------------------------------------------


def test_boxes_can_be_read_and_replaced_remotely(api: httpx.Client, dataset: Path) -> None:
    """This is what lets review happen from the laptop against a node."""
    dataset_id = register(api, dataset)
    path = f"/datasets/{dataset_id}/items/punch_001/boxes"

    assert api.get(path).json() == {
        "stem": "punch_001",
        "filename": "punch_001.jpg",
        "boxes": [],
        "reviewed": False,
    }

    response = api.put(path, json={"boxes": [{"x": 10, "y": 12, "w": 30, "h": 40}], "reviewed": True})
    assert response.status_code == 200

    stored = api.get(path).json()
    assert stored["reviewed"] is True
    assert stored["boxes"][0]["src"] == "manual", "a box from a client is one a human drew"
    assert paths.boxes_file(dataset).is_file()


def test_a_malformed_box_is_refused(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    response = api.put(
        f"/datasets/{dataset_id}/items/punch_001/boxes", json={"boxes": [{"x": 1, "y": 2}]}
    )
    assert response.status_code == 400


def test_review_progress(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    api.put(
        f"/datasets/{dataset_id}/items/punch_001/boxes",
        json={"boxes": [{"x": 1, "y": 1, "w": 5, "h": 5}], "reviewed": True},
    )
    progress = api.get(f"/datasets/{dataset_id}/review").json()
    assert progress["reviewed"] == 1 and progress["total"] == 4
    assert not progress["complete"]


# --------------------------------------------------------------------------
# operations
# --------------------------------------------------------------------------


def test_an_unknown_operation_is_404(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    assert api.post(f"/datasets/{dataset_id}/ops/reticulate", json={}).status_code == 404


def test_resize_over_the_api(api: httpx.Client, dataset_dir: Path, api_root: Path) -> None:
    make_image(dataset_dir / "wide.jpg", size=(1600, 800))
    dataset_id = register(api, dataset_dir)

    task = run_op(api, dataset_id, "resize", size=800)

    assert task["status"] == "done"
    assert task["result"]["processed"] == 1
    from fluxkrea.core.imaging import read_size

    assert read_size(dataset_dir / "wide.jpg").longest == 800


def test_rename_over_the_api_moves_the_bundle(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)

    task = run_op(api, dataset_id, "rename", prefix="kick")

    assert task["status"] == "done"
    assert task["result"]["renamed"] == 4
    assert (dataset / "kick_001.txt").is_file()
    assert (paths.masks_dir(dataset) / "kick_001.png").is_file()


def test_rename_without_a_prefix_is_a_400(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    assert api.post(f"/datasets/{dataset_id}/ops/rename", json={}).status_code == 400


def test_augment_over_the_api(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)

    task = run_op(api, dataset_id, "augment", transforms=["flip_horizontal"])

    assert task["result"]["created"] == 4
    assert (paths.masks_dir(dataset) / "punch_001_flipHor.png").is_file()


def test_a_failing_operation_reports_rather_than_500s(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    assert api.post(f"/datasets/{dataset_id}/ops/resize", json={"size": 0}).status_code == 400


def test_operations_are_listed_and_cancellable(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    run_op(api, dataset_id, "validate")

    tasks = api.get("/tasks", params={"dataset": dataset_id}).json()["tasks"]
    assert tasks and tasks[0]["kind"] == "dataset.validate"

    # A finished task cannot be cancelled, and says so rather than pretending.
    assert api.delete(f"/tasks/{tasks[0]['id']}").json()["cancelling"] is False


def test_unknown_task_is_404(api: httpx.Client) -> None:
    assert api.get("/tasks/nope").status_code == 404


# --------------------------------------------------------------------------
# masking, over the wire
# --------------------------------------------------------------------------


def test_masking_end_to_end_over_the_api(api: httpx.Client, dataset: Path) -> None:
    """Draw boxes remotely, export remotely, validate remotely."""
    dataset_id = register(api, dataset)

    for stem in ("punch_001", "punch_002", "punch_003", "punch_004"):
        api.put(
            f"/datasets/{dataset_id}/items/{stem}/boxes",
            json={"boxes": [{"x": 5, "y": 5, "w": 15, "h": 12}], "reviewed": True},
        )

    task = run_op(api, dataset_id, "mask", detect=False, expand=1.4, feather=0)

    assert task["status"] == "done", task
    assert task["result"]["written"] == 4
    assert api.get(
        f"/datasets/{dataset_id}/validate", params={"require_masks": True}
    ).json()["ok"] is True


def test_export_refuses_unreviewed_over_the_api(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    task = run_op(api, dataset_id, "mask", detect=False)

    assert task["status"] == "failed"
    assert len(task["result"]["refused"]) == 4


def test_mask_with_a_missing_detector_is_422(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    response = api.post(f"/datasets/{dataset_id}/ops/mask", json={"detector": "insightface"})
    assert response.status_code == 422


# --------------------------------------------------------------------------
# sync
# --------------------------------------------------------------------------


def test_manifest(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    payload = api.get(f"/datasets/{dataset_id}/manifest").json()

    paths_seen = {e["path"] for e in payload["entries"]}
    assert "punch_001.jpg" in paths_seen
    assert "masks/punch_001.png" in paths_seen, "masks must be part of the manifest"
    assert all("/" not in p or p.startswith(("masks/", "preview/")) for p in paths_seen)
    assert all(e["digest"] for e in payload["entries"])


def test_manifest_sidecars_only_is_the_cheap_loop(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    full = api.get(f"/datasets/{dataset_id}/manifest").json()
    small = api.get(f"/datasets/{dataset_id}/manifest", params={"sidecars_only": True}).json()

    assert small["bytes"] < full["bytes"]
    assert not any(e["path"].endswith(".jpg") for e in small["entries"])
    assert any(e["path"].startswith("masks/") for e in small["entries"])


def test_quick_manifest_skips_digests(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    payload = api.get(f"/datasets/{dataset_id}/manifest", params={"digests": False}).json()
    assert not any(e.get("digest") for e in payload["entries"])


def test_export_import_round_trip(api: httpx.Client, dataset: Path, tmp_path: Path) -> None:
    source_id = register(api, dataset)
    tar = api.get(f"/datasets/{source_id}/export").content
    assert tar

    target = tmp_path / "copy"
    target_id = register(api, target if target.mkdir() is None else target)

    response = api.post(f"/datasets/{target_id}/import", content=tar)
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] and payload["files"] >= 9

    assert (target / "punch_001.jpg").read_bytes() == (dataset / "punch_001.jpg").read_bytes()
    assert (paths.masks_dir(target) / "punch_001.png").is_file()


def test_import_refuses_a_non_tar(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    assert api.post(f"/datasets/{dataset_id}/import", content=b"not a tar").status_code == 400


# --------------------------------------------------------------------------
# filesystem browsing
# --------------------------------------------------------------------------


def test_browse_is_scoped_to_the_roots(api: httpx.Client, tmp_path: Path, dataset: Path) -> None:
    """There is no native file picker in a browser; this replaces it."""
    listing = api.get("/fs/browse").json()
    assert listing["roots"] == [tmp_path.as_posix()]

    inside = api.get("/fs/browse", params={"path": tmp_path.as_posix()}).json()
    names = {entry["name"] for entry in inside["entries"]}
    assert "poses" in names
    assert next(e for e in inside["entries"] if e["name"] == "poses")["images"] == 4

    outside = api.get("/fs/browse", params={"path": tmp_path.parent.as_posix()})
    assert outside.status_code == 403


def test_browse_marks_registered_folders(api: httpx.Client, tmp_path: Path, dataset: Path) -> None:
    register(api, dataset)
    entries = api.get("/fs/browse", params={"path": tmp_path.as_posix()}).json()["entries"]
    assert next(e for e in entries if e["name"] == "poses")["dataset_id"] == "poses"


@pytest.fixture
def api_root(tmp_path: Path) -> Path:
    return tmp_path
