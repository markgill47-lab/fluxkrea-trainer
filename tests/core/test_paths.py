"""Path resolution, including the cross-platform rules from doc 06."""

from __future__ import annotations

from pathlib import Path

import pytest

from fluxkrea.core import paths


def test_home_override_relocates_every_location(isolated_env: Path) -> None:
    for location in paths.describe().values():
        if location == str(paths.assets_dir()):
            continue  # vendored assets ship with the source, not the profile
        assert str(isolated_env) in location, location


def test_individual_overrides_beat_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FLUXKREA_CONFIG_DIR", str(tmp_path / "elsewhere"))
    assert paths.config_dir() == tmp_path / "elsewhere"
    assert paths.config_file() == tmp_path / "elsewhere" / "config.toml"


def test_blank_override_is_ignored(monkeypatch: pytest.MonkeyPatch, isolated_env: Path) -> None:
    monkeypatch.setenv("FLUXKREA_CONFIG_DIR", "   ")
    assert paths.config_dir() == isolated_env / "config"


def test_no_backslash_or_drive_letter_assumptions() -> None:
    """Doc 06: no separator or drive-letter literals anywhere in path handling."""
    source = (Path(paths.__file__)).read_text(encoding="utf-8")
    assert "\\\\" not in source
    assert ":\\" not in source
    for line in source.splitlines():
        assert "C:" not in line and "D:" not in line, line


def test_expand_handles_user_and_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FK_TEST_ROOT", str(tmp_path))
    assert paths.expand("$FK_TEST_ROOT/poses") == tmp_path / "poses"
    assert paths.expand("~").is_absolute()


def test_expand_does_not_require_existence(tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "deeper"
    assert paths.expand(missing) == missing


def test_is_within_scopes_browsing(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    (root / "poses").mkdir(parents=True)
    assert paths.is_within(root / "poses", root)
    assert paths.is_within(root, root)
    assert not paths.is_within(tmp_path / "elsewhere", root)


def test_is_within_rejects_dotdot_escape(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    root.mkdir()
    assert not paths.is_within(root / ".." / "secrets", root)


def test_dataset_layout_is_defined_once(tmp_path: Path) -> None:
    assert paths.masks_dir(tmp_path).name == "masks"
    assert paths.preview_dir(tmp_path).name == "preview"
    assert paths.boxes_file(tmp_path).parent == tmp_path
    assert list(paths.managed_dirs(tmp_path)) == [tmp_path / "masks", tmp_path / "preview"]


def test_ensure_dir_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b"
    assert paths.ensure_dir(target) == target
    assert paths.ensure_dir(target).is_dir()


def test_xdg_layout_on_linux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The Linux branch is exercised from Windows and vice versa.

    The fleet is Linux and the desk is Windows; neither platform may be
    the only one whose layout is ever executed.
    """
    monkeypatch.setattr("fluxkrea.core.paths.sys.platform", "linux")
    monkeypatch.delenv("FLUXKREA_HOME", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    assert paths.config_dir() == tmp_path / "cfg" / "fluxkrea"
    assert paths.data_dir() == tmp_path / "data" / "fluxkrea"
    assert paths.state_dir() == tmp_path / "state" / "fluxkrea"
    assert paths.cache_dir() == tmp_path / "cache" / "fluxkrea"


def test_windows_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("fluxkrea.core.paths.sys.platform", "win32")
    monkeypatch.delenv("FLUXKREA_HOME", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))

    assert paths.config_dir() == tmp_path / "Roaming" / "FluxKrea"
    assert paths.data_dir() == tmp_path / "Local" / "FluxKrea"
    assert paths.state_dir() == tmp_path / "Local" / "FluxKrea" / "State"


def test_xdg_falls_back_to_home_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fluxkrea.core.paths.sys.platform", "linux")
    monkeypatch.delenv("FLUXKREA_HOME", raising=False)
    for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    assert paths.config_dir() == Path.home() / ".config" / "fluxkrea"
