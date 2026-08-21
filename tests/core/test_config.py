"""Config precedence, coercion, and the no-secrets-in-the-file rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from fluxkrea.core import config, paths


def write_config(tmp_path: Path, text: str) -> Path:
    target = tmp_path / "config.toml"
    target.write_text(text, encoding="utf-8")
    return target


def test_defaults_load_with_no_file() -> None:
    cfg = config.load()
    assert cfg.daemon.host == "127.0.0.1"
    assert cfg.mask.expand == 1.6
    assert cfg.source is None
    assert cfg.validate() == []


def test_precedence_file_then_env_then_flags(tmp_path: Path) -> None:
    target = write_config(tmp_path, "[daemon]\nport = 9000\n[mask]\nexpand = 2.0\n")

    from_file = config.load(target, env={})
    assert from_file.daemon.port == 9000
    assert from_file.mask.expand == 2.0

    with_env = config.load(target, env={"FLUXKREA_DAEMON_PORT": "9100"})
    assert with_env.daemon.port == 9100
    assert with_env.mask.expand == 2.0

    with_flags = config.load(
        target, env={"FLUXKREA_DAEMON_PORT": "9100"}, overrides={"daemon.port": 9200}
    )
    assert with_flags.daemon.port == 9200


def test_unset_flag_does_not_clobber(tmp_path: Path) -> None:
    target = write_config(tmp_path, "[daemon]\nport = 9000\n")
    cfg = config.load(target, env={}, overrides={"daemon.port": None})
    assert cfg.daemon.port == 9000


def test_secret_in_file_is_a_hard_error(tmp_path: Path) -> None:
    target = write_config(tmp_path, '[captioner]\napi_key = "sk-should-not-be-here"\n')
    with pytest.raises(config.ConfigError, match="looks like a secret"):
        config.load(target, env={})


@pytest.mark.parametrize("key", ["api_key", "token", "client_secret", "password", "credentials"])
def test_every_secret_shape_is_rejected(tmp_path: Path, key: str) -> None:
    target = write_config(tmp_path, f'[daemon]\n{key} = "x"\n')
    with pytest.raises(config.ConfigError):
        config.load(target, env={})


def test_unknown_section_or_setting_raises(tmp_path: Path) -> None:
    with pytest.raises(config.ConfigError, match="unknown section"):
        config.load(write_config(tmp_path, "[nope]\nx = 1\n"), env={})
    with pytest.raises(config.ConfigError, match="unknown setting"):
        config.load(write_config(tmp_path, "[mask]\nexpandd = 1.0\n"), env={})


def test_missing_explicit_file_raises_but_missing_default_does_not(tmp_path: Path) -> None:
    with pytest.raises(config.ConfigError, match="not found"):
        config.load(tmp_path / "absent.toml")
    assert config.load().daemon.port == 8471


def test_env_coerces_types() -> None:
    cfg = config.load(
        env={
            "FLUXKREA_DAEMON_PORT": "8500",
            "FLUXKREA_MASK_EXPAND": "1.85",
            "FLUXKREA_MASK_INVERT": "yes",
            "FLUXKREA_DATASET_MIN_RESOLUTION": "768",
            "FLUXKREA_LOG_LEVEL": "debug",
        }
    )
    assert cfg.daemon.port == 8500
    assert cfg.mask.expand == pytest.approx(1.85)
    assert cfg.mask.invert is True
    assert cfg.dataset.min_resolution == 768
    assert cfg.log_level == "debug"


def test_env_list_uses_pathsep_and_expands(tmp_path: Path) -> None:
    import os

    raw = os.pathsep.join([str(tmp_path / "a"), str(tmp_path / "b")])
    cfg = config.load(env={"FLUXKREA_DATASET_ROOTS": raw})
    assert cfg.dataset.roots == [tmp_path / "a", tmp_path / "b"]


def test_optional_path_stays_none_and_expands_when_set(tmp_path: Path) -> None:
    cfg = config.load()
    assert cfg.backends.aitoolkit_path is None
    cfg.set("backends.aitoolkit_path", str(tmp_path / "ai-toolkit"))
    assert cfg.backends.aitoolkit_path == tmp_path / "ai-toolkit"


def test_bad_value_is_reported_not_guessed() -> None:
    with pytest.raises(config.ConfigError, match="expects an integer"):
        config.load(env={"FLUXKREA_DAEMON_PORT": "eight thousand"})


def test_round_trip_through_disk(tmp_path: Path) -> None:
    cfg = config.load()
    cfg.set("mask.expand", 1.9)
    cfg.set("dataset.roots", [str(tmp_path)])
    saved = cfg.save(tmp_path / "out.toml")

    reloaded = config.load(saved, env={})
    assert reloaded.mask.expand == 1.9
    assert reloaded.dataset.roots == [tmp_path]
    assert reloaded.source == saved


def test_saved_file_uses_posix_separators(tmp_path: Path) -> None:
    """The same config file is read on the Windows desk and the Linux fleet."""
    cfg = config.load()
    cfg.set("dataset.roots", [str(tmp_path / "poses")])
    text = cfg.save(tmp_path / "out.toml").read_text(encoding="utf-8")
    assert "\\\\" not in text


def test_saved_file_never_contains_a_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUXKREA_CLAUDE_API_KEY", "sk-secret-value")
    target = config.load().save(tmp_path / "out.toml")
    assert "sk-secret-value" not in target.read_text(encoding="utf-8")

    # The strongest statement available: what was written loads again. A
    # secret in it would be a hard error on the way back in, so a clean
    # round trip *is* the assertion that none was written.
    config.load(path=target)


def test_a_string_setting_named_like_a_secret_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "sneaky.toml"
    target.write_text('[captioner]\napi_key = "sk-nope"\n', encoding="utf-8")
    with pytest.raises(config.ConfigError, match="looks like a secret"):
        config.load(path=target)


def test_a_number_named_after_a_secret_is_a_count_not_a_secret(tmp_path: Path) -> None:
    """``max_tokens`` contains "token" and is a length limit."""
    target = tmp_path / "counts.toml"
    target.write_text("[captioner]\nmax_tokens = 250\n", encoding="utf-8")
    assert config.load(path=target).captioner.max_tokens == 250


def test_secret_comes_from_env_with_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    assert config.secret("claude") is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-alias")
    assert config.secret("claude") == "from-alias"
    monkeypatch.setenv("FLUXKREA_CLAUDE_API_KEY", "from-primary")
    assert config.secret("claude") == "from-primary"


def test_validate_reports_problems_without_raising(tmp_path: Path) -> None:
    cfg = config.load()
    cfg.daemon.port = 70000
    cfg.mask.expand = 0.5
    cfg.mask.min_value = 5.0
    cfg.dataset.image_extensions = ["jpg"]
    cfg.dataset.roots = [tmp_path / "missing"]

    problems = cfg.validate()
    assert len(problems) == 5
    assert any("port" in p for p in problems)
    assert any("must start with a dot" in p for p in problems)


def test_non_loopback_bind_requires_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Doc 06: no silent open port on something that launches processes."""
    cfg = config.load()
    cfg.daemon.host = "0.0.0.0"  # noqa: S104 - the case under test
    assert any("FLUXKREA_TOKEN" in p for p in cfg.validate())

    monkeypatch.setenv("FLUXKREA_TOKEN", "a-token")
    assert cfg.validate() == []


def test_dotted_access(tmp_path: Path) -> None:
    cfg = config.load()
    assert cfg.get("mask.expand") == 1.6
    assert cfg.get("mask") is cfg.mask
    with pytest.raises(KeyError):
        cfg.set("mask", 1)
    with pytest.raises(KeyError):
        cfg.set("nope.thing", 1)


def test_default_file_location_is_honoured(isolated_env: Path) -> None:
    paths.ensure_dir(paths.config_dir())
    paths.config_file().write_text("[daemon]\nport = 8999\n", encoding="utf-8")
    cfg = config.load()
    assert cfg.daemon.port == 8999
    assert cfg.source == paths.config_file()
