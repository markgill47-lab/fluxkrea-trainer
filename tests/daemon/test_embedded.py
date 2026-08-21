"""The per-command daemon.

The CLI is an API client and the API owns the logic, so on a desktop with
nothing running the usual answer is a "local mode" that calls ``core``
directly - which is precisely the divergence doc 02 forbids. These tests
pin the alternative: the real daemon, on an ephemeral loopback port, for
the life of one command.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from fluxkrea.cli.client import Client
from fluxkrea.cli.embedded import EmbeddedDaemon
from fluxkrea.core.config import load


def test_it_starts_and_stops(tmp_path: Path) -> None:
    daemon = EmbeddedDaemon(load())
    url = daemon.start()

    assert url.startswith("http://127.0.0.1:")
    assert daemon.port > 0

    with Client.remote(url) as client:
        assert client.get("/health")["status"] == "ok"

    daemon.stop()
    with socket.socket() as probe:
        probe.settimeout(1)
        assert probe.connect_ex(("127.0.0.1", daemon.port)) != 0, "the port is still open"


def test_it_binds_loopback_only(tmp_path: Path) -> None:
    """Port 0 on 127.0.0.1: nothing reachable from off the machine."""
    daemon = EmbeddedDaemon(load())
    try:
        assert daemon.start().startswith("http://127.0.0.1:")
    finally:
        daemon.stop()


def test_the_client_picks_it_up_when_nothing_is_listening(tmp_path: Path) -> None:
    config = load()
    config.daemon.port = _free_port()

    with Client.for_config(config) as client:
        assert client.embedded
        assert client.where == "this machine"
        assert client.get("/health")["status"] == "ok"


def test_the_client_prefers_a_daemon_that_is_already_running(tmp_path: Path) -> None:
    """Otherwise two of them would fight over one dataset folder."""
    running = EmbeddedDaemon(load())
    url = running.start()
    try:
        config = load()
        config.daemon.port = running.port
        with Client.for_config(config) as client:
            assert not client.embedded
            assert client.base_url == url
    finally:
        running.stop()


def test_an_explicit_url_wins(tmp_path: Path) -> None:
    with Client.for_config(load(), "http://somewhere:9999") as client:
        assert not client.embedded
        assert client.base_url == "http://somewhere:9999"


def test_a_full_operation_runs_through_the_embedded_daemon(dataset: Path) -> None:
    """The point of the whole arrangement: no divergent local path."""
    config = load()
    config.dataset.min_resolution = 0
    config.daemon.port = _free_port()

    with Client.for_config(config) as client:
        registered = client.resolve(dataset.as_posix())
        task = client.post(
            f"/datasets/{registered['id']}/ops/validate",
            json_body={"require_masks": True},
        )
        final = client.watch(task["id"])

    assert final["status"] == "failed", "three of the four items have no mask"
    assert final["result"]["counts"]["missing_mask"] == 3


def test_a_failed_start_does_not_leave_a_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    daemon = EmbeddedDaemon(load())
    monkeypatch.setattr("fluxkrea.cli.embedded.STARTUP_TIMEOUT", 0.2)
    monkeypatch.setattr(
        "fluxkrea.daemon.app.create_app",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no app for you")),
    )

    before = threading.active_count()
    with pytest.raises(RuntimeError):
        daemon.start()
    assert threading.active_count() <= before + 1


def _free_port() -> int:
    """A port nothing is on, so the client falls through to embedded."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
