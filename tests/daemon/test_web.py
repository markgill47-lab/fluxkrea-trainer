"""Thumbnails and serving the client.

Both exist for the same reason: doc 02 chose a browser client because
"nothing to install on any machine you drive from - the client arrives
with the daemon", and doc 10 requires the daemon to shrink images before
they cross a tunnel. Neither claim holds without the code here.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from PIL import Image

from fluxkrea.core import paths
from fluxkrea.core.dataset import thumbs
from fluxkrea.daemon import web
from tests.conftest import make_image
from tests.daemon.conftest import register


# --------------------------------------------------------------------------
# thumbnails
# --------------------------------------------------------------------------


def test_a_thumbnail_is_much_smaller_than_its_source(dataset_dir: Path) -> None:
    """The whole justification: a 160px cell must not cost a full image."""
    source = make_image(dataset_dir / "big.jpg", size=(3000, 2000))

    target = thumbs.build(source, "poses", "big", 160)

    assert target.suffix == ".webp"
    assert target.stat().st_size < source.stat().st_size / 10
    # Doc 10's budget for a 160px thumbnail.
    assert target.stat().st_size < 12_000


def test_thumbnails_fit_inside_the_requested_box(dataset_dir: Path) -> None:
    source = make_image(dataset_dir / "wide.jpg", size=(2000, 500))
    with Image.open(thumbs.build(source, "poses", "wide", 160)) as thumb:
        assert max(thumb.size) <= 160
        assert thumb.width / thumb.height == pytest.approx(4, rel=0.05)


def test_thumbnails_are_cached_and_reused(dataset_dir: Path) -> None:
    source = make_image(dataset_dir / "a.jpg", size=(800, 600))

    first = thumbs.build(source, "poses", "a", 160)
    stamp = first.stat().st_mtime_ns
    second = thumbs.build(source, "poses", "a", 160)

    assert first == second
    assert second.stat().st_mtime_ns == stamp, "the thumbnail was regenerated"


def test_a_changed_source_gets_a_different_token_and_file(dataset_dir: Path) -> None:
    """This is what makes the URL self-busting: no invalidation logic."""
    source = make_image(dataset_dir / "a.jpg", size=(800, 600))
    before = thumbs.build(source, "poses", "a", 160)
    before_token = thumbs.token_for(source)

    make_image(dataset_dir / "a.jpg", size=(640, 480), colour=(10, 200, 30))
    after = thumbs.build(source, "poses", "a", 160)

    assert thumbs.token_for(source) != before_token
    assert after != before


def test_thumbnails_honour_exif_orientation(dataset_dir: Path) -> None:
    """A filmstrip cell that disagrees with the canvas is a bug report."""
    source = make_image(dataset_dir / "rotated.jpg", size=(400, 200), exif_orientation=6)
    with Image.open(thumbs.build(source, "poses", "rotated", 160)) as thumb:
        assert thumb.height > thumb.width


def test_thumbnails_live_outside_the_dataset_folder(dataset_dir: Path) -> None:
    """Dataset folders are training data and get rsynced; derived files
    do not belong in them."""
    source = make_image(dataset_dir / "a.jpg")
    target = thumbs.build(source, "poses", "a", 160)

    assert paths.cache_dir() in target.parents
    assert dataset_dir not in target.parents


def test_an_unsupported_size_is_refused(dataset_dir: Path) -> None:
    source = make_image(dataset_dir / "a.jpg")
    with pytest.raises(ValueError, match="must be one of"):
        thumbs.build(source, "poses", "a", 999)


def test_ensure_many_skips_a_corrupt_image(dataset_dir: Path) -> None:
    """One bad file should not stop the other 209."""
    good = make_image(dataset_dir / "good.jpg")
    bad = dataset_dir / "bad.jpg"
    bad.write_bytes(b"not an image")

    built = thumbs.ensure_many([(good, "good"), (bad, "bad")], "poses")
    assert built == 1


def test_purge(dataset_dir: Path) -> None:
    thumbs.build(make_image(dataset_dir / "a.jpg"), "poses", "a", 160)
    assert thumbs.purge("poses") == 1
    assert thumbs.purge("poses") == 0


# --------------------------------------------------------------------------
# the endpoint
# --------------------------------------------------------------------------


def test_the_thumb_endpoint_serves_webp(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    response = api.get(f"/datasets/{dataset_id}/items/punch_001/thumb")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert response.content[:4] == b"RIFF"


def test_a_token_makes_the_response_immutable(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    token = api.get(f"/datasets/{dataset_id}/items").json()["items"][0]["token"]

    with_token = api.get(f"/datasets/{dataset_id}/items/punch_001/thumb", params={"v": token})
    without = api.get(f"/datasets/{dataset_id}/items/punch_001/thumb")

    assert "immutable" in with_token.headers["cache-control"]
    assert "no-cache" in without.headers["cache-control"]


def test_items_carry_a_token_for_the_client_to_use(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    items = api.get(f"/datasets/{dataset_id}/items").json()["items"]
    assert all(item["token"] for item in items)


def test_a_bad_size_is_a_400(api: httpx.Client, dataset: Path) -> None:
    dataset_id = register(api, dataset)
    assert (
        api.get(f"/datasets/{dataset_id}/items/punch_001/thumb", params={"size": 99}).status_code
        == 400
    )


def test_an_unreadable_image_is_a_404_on_that_cell(api: httpx.Client, dataset: Path) -> None:
    (dataset / "broken.jpg").write_bytes(b"not an image")
    dataset_id = register(api, dataset)

    response = api.get(f"/datasets/{dataset_id}/items/broken/thumb")
    assert response.status_code == 404


# --------------------------------------------------------------------------
# serving the client
# --------------------------------------------------------------------------


def test_no_build_means_the_api_still_works(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A daemon with no client build is perfectly usable from fk and curl."""
    monkeypatch.setenv("FLUXKREA_WEB_DIR", str(tmp_path / "nothing-here"))
    assert web.find_client() is None


def test_an_explicit_directory_is_used(tmp_path: Path) -> None:
    root = tmp_path / "dist"
    root.mkdir()
    (root / "index.html").write_text("<!doctype html>", encoding="utf-8")
    assert web.find_client(root) == root


def test_a_directory_without_an_index_is_not_a_client(tmp_path: Path) -> None:
    empty = tmp_path / "dist"
    empty.mkdir()
    assert web.find_client(empty) is None


def test_the_client_is_served_and_the_api_still_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config
) -> None:
    from starlette.testclient import TestClient

    from fluxkrea.daemon.app import API, create_app
    from fluxkrea.daemon.registry import Registry
    from fluxkrea.daemon.state import State

    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>client</title>", encoding="utf-8")
    (root / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    monkeypatch.setenv("FLUXKREA_WEB_DIR", str(root))

    state = State(config=config, registry=Registry(file=tmp_path / "registry.json"))
    try:
        with TestClient(create_app(state=state)) as client:
            assert "client" in client.get("/").text
            # A client-side route falls back to the shell, not a 404.
            assert client.get("/review/poses").status_code == 200
            # Assets are cached hard; the shell is not.
            assert "immutable" in client.get("/assets/app.js").headers["cache-control"]
            assert "no-cache" in client.get("/").headers["cache-control"]
            # The API is mounted first and still answers.
            assert client.get(f"{API}/health").json()["status"] == "ok"
            # A mistyped API path is a 404, not the app shell with a 200 -
            # which a client would see as a broken JSON parse.
            assert client.get(f"{API}/nope").status_code == 404
    finally:
        state.shutdown()
