"""Daemon fixtures.

The API tests run against a real ASGI app, not against the route functions
directly - the serialisation and the status codes are half of what a
contract test is for. ``TestClient`` is starlette's synchronous bridge and
belongs here; the CLI's embedded mode uses a real server instead, because
that is production code.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from fluxkrea.core.config import Config, load
from fluxkrea.daemon.app import API, create_app
from fluxkrea.daemon.registry import Registry
from fluxkrea.daemon.state import State


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """A config scoped to the test's temp directory, as a node would be."""
    cfg = load()
    cfg.dataset.roots = [tmp_path]
    cfg.dataset.min_resolution = 0
    cfg.daemon.node_name = "test-node"
    return cfg


@pytest.fixture
def state(config: Config, tmp_path: Path) -> State:
    return State(config=config, registry=Registry(file=tmp_path / "registry.json"))


@pytest.fixture
def api(state: State) -> Iterator[httpx.Client]:
    """An HTTP client bound to the app, with no socket in between."""
    with TestClient(create_app(state=state), base_url=f"http://node{API}") as client:
        client.app_state = state  # type: ignore[attr-defined]
        yield client
    state.shutdown()


def register(api: httpx.Client, path: Path, name: str | None = None) -> str:
    response = api.post("/datasets", json={"path": path.as_posix(), "name": name})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def run_op(api: httpx.Client, dataset_id: str, operation: str, **options) -> dict:
    """Submit an operation and wait for it, the way a synchronous client does."""
    response = api.post(f"/datasets/{dataset_id}/ops/{operation}", json=options)
    assert response.status_code == 202, response.text
    task_id = response.json()["id"]

    task = api.app_state.tasks.get(task_id)  # type: ignore[attr-defined]
    assert task is not None
    assert task.wait(30), f"{operation} did not finish"
    return api.get(f"/tasks/{task_id}").json()


# --------------------------------------------------------------------------
# real servers
# --------------------------------------------------------------------------


def _live(root: Path, registry_file: Path) -> Iterator:
    """A daemon on a real ephemeral port.

    The fleet client speaks plain HTTP to real nodes, so testing it needs a
    real socket - the same ``EmbeddedDaemon`` the CLI uses when nothing is
    listening locally.
    """
    from fluxkrea.cli.client import Client
    from fluxkrea.cli.embedded import EmbeddedDaemon

    cfg = load()
    cfg.dataset.roots = [root]
    cfg.dataset.min_resolution = 0

    node_state = State(config=cfg, registry=Registry(file=registry_file))
    daemon = EmbeddedDaemon(cfg, node_state)
    url = daemon.start()

    client = Client.remote(url)
    try:
        yield client
    finally:
        client.close()
        daemon.stop()
        node_state.shutdown()


@pytest.fixture
def live_node(tmp_path: Path) -> Iterator:
    yield from _live(tmp_path, tmp_path / "live-registry.json")


@pytest.fixture
def second_live_node(tmp_path: Path) -> Iterator:
    yield from _live(tmp_path, tmp_path / "second-registry.json")
