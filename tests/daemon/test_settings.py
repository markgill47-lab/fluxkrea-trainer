"""The settings endpoints, and the two things they refuse.

Doc 06 puts the daemon on loopback and scopes every path to the configured
roots. A ``PUT /config`` that could rewrite either would undo both from a
browser, so those refusals are contract, not preference - and the tests
below are what keeps them that way through a refactor.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from fluxkrea.core.config import Config


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def test_config_comes_back_whole(api: httpx.Client) -> None:
    body = api.get("/config").json()
    for section in ("dataset", "mask", "captioner", "daemon", "backends"):
        assert section in body
    assert "read_only" in body


def test_the_config_carries_no_secrets(api: httpx.Client) -> None:
    """The rule that lets config.toml be committed and shared."""
    text = api.get("/config").text.lower()
    for hint in ("api_key", "apikey", "token=", "password"):
        assert hint not in text


def test_secrets_report_presence_and_never_values(
    api: httpx.Client, monkeypatch
) -> None:
    monkeypatch.setenv("FLUXKREA_CLAUDE_API_KEY", "sk-ant-not-a-real-key")
    body = api.get("/config/secrets").json()

    claude = next(s for s in body["secrets"] if s["name"] == "claude")
    assert claude["found"] is True
    assert "FLUXKREA_CLAUDE_API_KEY" in claude["env"]
    assert "sk-ant-not-a-real-key" not in api.get("/config/secrets").text


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def test_a_setting_can_be_changed_and_is_written(api: httpx.Client, tmp_path: Path) -> None:
    target = tmp_path / "written.toml"
    api.app_state.config.source = target  # type: ignore[attr-defined]

    response = api.put("/config", json={"set": {"captioner.ollama_model": "llava"}})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["captioner"]["ollama_model"] == "llava"
    assert body["changed"] == ["captioner.ollama_model"]
    assert Path(body["written"]).is_file()
    assert "llava" in Path(body["written"]).read_text(encoding="utf-8")


def test_the_running_daemon_sees_the_change(api: httpx.Client) -> None:
    api.put("/config", json={"set": {"captioner.provider": "claude"}})
    assert api.app_state.config.captioner.provider == "claude"  # type: ignore[attr-defined]
    assert api.get("/captioners").json()["configured"] == "claude"


def test_several_settings_apply_together(api: httpx.Client) -> None:
    response = api.put(
        "/config",
        json={"set": {"mask.feather": 12, "mask.expand": 1.8, "captioner.prefix": "mara"}},
    )
    body = response.json()
    assert body["mask"]["feather"] == 12
    assert body["mask"]["expand"] == 1.8
    assert body["captioner"]["prefix"] == "mara"


def test_a_bare_map_works_without_the_set_wrapper(api: httpx.Client) -> None:
    body = api.put("/config", json={"captioner.prefix": "olympus"}).json()
    assert body["captioner"]["prefix"] == "olympus"


def test_an_empty_body_is_a_bad_request(api: httpx.Client) -> None:
    assert api.put("/config", json={}).status_code == 400


def test_an_unknown_setting_is_refused_by_name(api: httpx.Client) -> None:
    response = api.put("/config", json={"set": {"captioner.wibble": 1}})
    assert response.status_code == 422
    assert "captioner.wibble" in response.json()["error"]


def test_a_setting_that_fails_validation_is_refused(api: httpx.Client) -> None:
    response = api.put("/config", json={"set": {"mask.expand": 0.2}})
    assert response.status_code == 422
    assert "expand" in response.json()["error"]


def test_a_rejected_change_leaves_the_daemon_on_its_old_config(api: httpx.Client) -> None:
    """Validated against a copy, so a refusal is not a half-applied config."""
    before = api.app_state.config.mask.feather  # type: ignore[attr-defined]
    api.put("/config", json={"set": {"mask.feather": 4, "mask.expand": 0.1}})
    assert api.app_state.config.mask.feather == before  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# the refusals
# --------------------------------------------------------------------------


def test_the_daemon_section_is_not_editable_over_http(api: httpx.Client) -> None:
    """Changing the bind address from a browser would widen the API's reach."""
    response = api.put("/config", json={"set": {"daemon.host": "0.0.0.0"}})
    assert response.status_code == 403
    assert api.app_state.config.daemon.host == "127.0.0.1"  # type: ignore[attr-defined]


def test_dataset_roots_are_not_editable_over_http(api: httpx.Client, tmp_path: Path) -> None:
    """roots is the allow-list every path check is measured against."""
    before = list(api.app_state.config.dataset.roots)  # type: ignore[attr-defined]
    response = api.put("/config", json={"set": {"dataset.roots": ["C:/"]}})
    assert response.status_code == 403
    assert "config.toml" in response.json()["error"]
    assert api.app_state.config.dataset.roots == before  # type: ignore[attr-defined]


def test_the_rest_of_the_dataset_section_still_is(api: httpx.Client) -> None:
    """One locked key must not lock its whole section."""
    body = api.put("/config", json={"set": {"dataset.caption_ext": ".caption"}}).json()
    assert body["dataset"]["caption_ext"] == ".caption"


def test_a_section_name_alone_is_refused(api: httpx.Client) -> None:
    assert api.put("/config", json={"set": {"captioner": {}}}).status_code == 400


def test_the_locked_settings_are_advertised(api: httpx.Client) -> None:
    """So the UI can say why a field is read-only rather than just failing."""
    read_only = api.get("/config").json()["read_only"]
    assert "daemon.*" in read_only
    assert "dataset.roots" in read_only


# --------------------------------------------------------------------------
# captioners
# --------------------------------------------------------------------------


def test_the_captioner_list_names_what_is_configured(api: httpx.Client) -> None:
    body = api.get("/captioners").json()
    names = {c["name"] for c in body["captioners"]}
    assert {"ollama", "joycaption", "claude"} <= names
    assert body["configured"] == "ollama"
    assert all("label" in c and "available" in c for c in body["captioners"])


def test_the_node_reports_captioners_without_probing_them(api: httpx.Client) -> None:
    assert "ollama" in api.get("/node").json()["captioners"]


def test_probing_a_stopped_backend_is_an_answer_not_an_error(api: httpx.Client) -> None:
    """The settings screen renders the message either way, so 200 it is."""
    response = api.post(
        "/captioners/test",
        json={"provider": "ollama", "url": "http://127.0.0.1:1", "timeout": 1},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "ollama serve" in body["message"]


def test_probing_an_unknown_provider_says_which_exist(api: httpx.Client) -> None:
    body = api.post("/captioners/test", json={"provider": "blip2"}).json()
    assert body["ok"] is False
    assert "ollama" in body["message"]


def test_captioning_is_an_offered_operation(api: httpx.Client) -> None:
    """Reachable by the same route as every other dataset operation."""
    from fluxkrea.daemon.routes.datasets import OPS

    assert "caption" in OPS


def test_a_caption_op_with_a_dead_backend_fails_the_task_not_the_request(
    api: httpx.Client, tmp_path: Path
) -> None:
    from PIL import Image

    from tests.daemon.conftest import register, run_op

    root = tmp_path / "set"
    root.mkdir()
    Image.new("RGB", (64, 64)).save(root / "a.png")
    dataset_id = register(api, root)

    # The request is accepted; the unreachable backend surfaces as a failed
    # task with a reason, not as an HTTP error on the submission.
    final = run_op(api, dataset_id, "caption", url="http://127.0.0.1:1", timeout=1)
    assert final["status"] == "failed"
    assert final["result"]["ok"] is False
    assert "ollama serve" in final["result"]["aborted"]


def test_config_defaults_keep_captioning_local(api: httpx.Client) -> None:
    """Nothing leaves the node unless someone changes this on purpose."""
    assert Config().captioner.provider == "ollama"
