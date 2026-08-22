"""The API is remote code execution scoped to the node. These are the fences."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from fluxkrea.core.config import Config, load
from fluxkrea.daemon.app import API, create_app
from fluxkrea.daemon.security import (
    Denied,
    check_bind,
    check_path,
    check_token,
    is_loopback,
    lan_addresses,
)
from fluxkrea.daemon.state import State


def client_for(config: Config) -> TestClient:
    return TestClient(create_app(state=State(config=config)), base_url=f"http://node{API}")


# --------------------------------------------------------------------------
# binding
# --------------------------------------------------------------------------


def test_loopback_needs_no_token() -> None:
    config = load()
    assert is_loopback(config.daemon.host)
    check_bind(config)  # must not raise


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "::"])  # noqa: S104
def test_binding_wider_without_a_token_is_refused(host: str) -> None:
    """No silent open port on something that launches processes."""
    config = load()
    config.daemon.host = host
    with pytest.raises(Denied, match="without a token"):
        check_bind(config)


def test_binding_wider_with_a_token_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUXKREA_TOKEN", "a-real-token")
    config = load()
    config.daemon.host = "0.0.0.0"  # noqa: S104 - the case under test
    check_bind(config)


# --------------------------------------------------------------------------
# lab mode
# --------------------------------------------------------------------------


def test_lab_mode_permits_an_open_bind_with_no_token(tmp_path: Path) -> None:
    """A classroom on one LAN node. The trade is stated, not stumbled into."""
    config = load()
    config.daemon.host = "0.0.0.0"  # noqa: S104 - the case under test
    config.daemon.lab_mode = True
    config.dataset.roots = [tmp_path]
    check_bind(config)


def test_lab_mode_without_roots_is_refused(tmp_path: Path) -> None:
    """Lab mode drops the token, so the roots are the only remaining fence."""
    config = load()
    config.daemon.host = "0.0.0.0"  # noqa: S104 - the case under test
    config.daemon.lab_mode = True
    config.dataset.roots = []
    with pytest.raises(Denied, match="dataset.roots"):
        check_bind(config)


def test_lab_mode_is_off_by_default() -> None:
    assert load().daemon.lab_mode is False


def test_lab_mode_does_not_relax_the_path_check(tmp_path: Path) -> None:
    """The roots still scope every path, token or no token."""
    config = load()
    config.daemon.lab_mode = True
    config.dataset.roots = [tmp_path]
    with pytest.raises(Denied, match="outside the configured dataset roots"):
        check_path(config, tmp_path.parent / "somewhere-else")


def test_lab_mode_is_not_writable_over_the_api(api: httpx.Client) -> None:
    """Turning it on is a decision made with a shell on the node."""
    response = api.put("/config", json={"set": {"daemon.lab_mode": True}})
    assert response.status_code == 403
    assert "config.toml" in response.json()["error"]


def test_a_lab_node_reports_its_own_urls() -> None:
    """Printed at startup, because "log in here" is the whole instruction."""
    urls = lan_addresses(8471)
    assert all(url.startswith("http://") and url.endswith(":8471") for url in urls)
    assert not any(url.startswith("http://127.") for url in urls)


# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------


def test_no_token_configured_means_no_check() -> None:
    check_token(load(), None)


def test_a_configured_token_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUXKREA_TOKEN", "secret-value")
    config = load()

    check_token(config, "secret-value")
    for wrong in (None, "", "nearly-secret-value"):
        with pytest.raises(Denied) as exc:
            check_token(config, wrong)
        assert exc.value.status == 401


def test_requests_without_a_token_are_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUXKREA_TOKEN", "secret-value")
    with client_for(load()) as api:
        assert api.get("/datasets").status_code == 401
        assert api.get("/datasets", headers={"x-fluxkrea-token": "secret-value"}).status_code == 200
        assert api.get(
            "/datasets", headers={"authorization": "Bearer secret-value"}
        ).status_code == 200


def test_health_stays_open_so_a_probe_can_reach_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUXKREA_TOKEN", "secret-value")
    with client_for(load()) as api:
        assert api.get("/health").status_code == 200


# --------------------------------------------------------------------------
# path scoping
# --------------------------------------------------------------------------


def test_paths_are_scoped_to_the_configured_roots(tmp_path: Path) -> None:
    config = load()
    config.dataset.roots = [tmp_path / "datasets"]
    (tmp_path / "datasets" / "poses").mkdir(parents=True)

    assert check_path(config, tmp_path / "datasets" / "poses").is_dir()
    with pytest.raises(Denied):
        check_path(config, tmp_path / "elsewhere")


def test_dotdot_cannot_escape_a_root(tmp_path: Path) -> None:
    config = load()
    root = tmp_path / "datasets"
    root.mkdir()
    config.dataset.roots = [root]

    with pytest.raises(Denied):
        check_path(config, root / ".." / "secrets")


def test_no_roots_means_no_restriction(tmp_path: Path) -> None:
    """Only reasonable because the default bind is loopback."""
    config = load()
    assert config.dataset.roots == []
    assert check_path(config, tmp_path) == tmp_path


def test_the_import_endpoint_cannot_be_used_to_write_outside(tmp_path: Path) -> None:
    """A tar is a list of paths chosen by whoever built it."""
    import io
    import tarfile

    from fluxkrea.core.dataset.archive import extract

    target = tmp_path / "dataset"
    target.mkdir()
    victim = tmp_path / "victim.txt"

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in ("../victim.txt", "/etc/passwd", "deep/nested/thing.txt"):
            info = tarfile.TarInfo(name=name)
            info.size = 4
            archive.addfile(info, io.BytesIO(b"evil"))
    buffer.seek(0)

    result = extract(target, buffer)

    assert result.files == 0
    assert len(result.skipped) == 3
    assert not victim.exists()
    assert not (target / "deep").exists()


def test_symlinks_in_a_tar_are_refused(tmp_path: Path) -> None:
    import io
    import tarfile

    from fluxkrea.core.dataset.archive import extract

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo(name="link.txt")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    buffer.seek(0)

    result = extract(tmp_path, buffer)
    assert result.files == 0
    assert "not a regular file" in result.skipped[0]
