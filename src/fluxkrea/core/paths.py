"""Every path the application knows, resolved in one place.

Rules this module exists to enforce (doc 06, "cross-platform rules"):

* ``pathlib`` throughout. No backslash literals, no drive-letter
  assumptions, no ``os.path.join`` string surgery.
* Config, data, cache and state live in OS-appropriate locations, never
  next to the source tree.
* Every location is overridable by environment variable, so a test - or a
  second daemon on the same box - never touches the real user profile.

Everything here is a function rather than a module constant: the
environment is read at call time, which is what makes the overrides
usable from tests and from a systemd unit alike.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

APP_NAME = "fluxkrea"
APP_DIR_NAME = "FluxKrea"  # Windows convention: title case under AppData

#: Sidecar and output layout inside a dataset folder. One definition, so
#: the scanner, the mask exporter and the sync manifest cannot disagree.
MASKS_DIRNAME = "masks"
PREVIEW_DIRNAME = "preview"
BOXES_FILENAME = "face_boxes.json"
METADATA_FILENAME = "metadata.json"

CONFIG_FILENAME = "config.toml"
FLEET_FILENAME = "fleet.toml"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _env_path(name: str) -> Path | None:
    """Read an environment variable as a path, ignoring blanks."""
    raw = os.environ.get(name, "").strip()
    return expand(raw) if raw else None


def expand(value: str | os.PathLike[str]) -> Path:
    """Expand ``~`` and ``$VARS`` in a user-supplied path, without resolving.

    Kept separate from :func:`resolve` because a path that does not exist
    yet still needs expanding.
    """
    return Path(os.path.expandvars(os.fspath(value))).expanduser()


def resolve(value: str | os.PathLike[str]) -> Path:
    """Expand and fully resolve a user-supplied path."""
    return expand(value).resolve()


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if missing, and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_within(child: str | os.PathLike[str], parent: str | os.PathLike[str]) -> bool:
    """True if *child* is *parent* or lives underneath it.

    Used to scope filesystem browsing and dataset registration to the
    configured roots - the API is remote code execution otherwise (doc 06,
    "security"). Comparison is done on resolved paths so ``..`` and
    symlinks cannot escape.
    """
    try:
        resolve(child).relative_to(resolve(parent))
    except (ValueError, OSError):
        return False
    return True


#: Windows returns this from ``GetCurrentPackageFullName`` when the process
#: is not inside an app package. Anything else - including "buffer too
#: small" - means it is.
_APPMODEL_ERROR_NO_PACKAGE = 15700


def app_package() -> str | None:
    r"""The Windows app package this process runs inside, or ``None``.

    A process launched from inside an MSIX package - the terminal embedded
    in a packaged desktop app, for instance - gets a *private* view of
    ``%APPDATA%`` and ``%LOCALAPPDATA%``. The path strings are byte for byte
    what the host uses; the files behind them are not, and nothing outside
    the container can see the writes.

    That cost an afternoon. A daemon started that way read a config file it
    named as ``AppData\Roaming\FluxKrea\config.toml``, reported settings
    the file on disk did not contain, wrote changes that never appeared in
    it, and failed every training run with "backends.aitoolkit_path is not
    set" while that exact file had it set. Every observation was correct and
    the conclusion was impossible, because two files answered to one path.

    So: ask. It is one syscall, and the answer turns an impossible bug into
    a sentence.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        length = ctypes.c_uint32(0)
        rc = ctypes.windll.kernel32.GetCurrentPackageFullName(ctypes.byref(length), None)
        if rc == _APPMODEL_ERROR_NO_PACKAGE:
            return None
        buffer = ctypes.create_unicode_buffer(max(length.value, 512))
        length = ctypes.c_uint32(len(buffer))
        rc = ctypes.windll.kernel32.GetCurrentPackageFullName(ctypes.byref(length), buffer)
        if rc == _APPMODEL_ERROR_NO_PACKAGE:
            return None
        return buffer.value or "unknown package"
    except (AttributeError, OSError):
        # Pre-Windows-8 has no appmodel API at all, which means no packages.
        return None


def home() -> Path:
    """The user's home directory, honouring ``FLUXKREA_HOME``.

    ``FLUXKREA_HOME`` relocates *everything* - config, data, cache, state -
    under one directory. That is the portable-install case and the test
    case; the individual overrides below are for finer control.
    """
    override = _env_path("FLUXKREA_HOME")
    return override if override else Path.home()


def _under_home(*parts: str) -> Path | None:
    override = _env_path("FLUXKREA_HOME")
    return override.joinpath(*parts) if override else None


# --------------------------------------------------------------------------
# per-user locations
# --------------------------------------------------------------------------


def config_dir() -> Path:
    """Where ``config.toml`` and ``fleet.toml`` live."""
    if (p := _env_path("FLUXKREA_CONFIG_DIR")) is not None:
        return p
    if (p := _under_home("config")) is not None:
        return p
    if sys.platform == "win32":
        return _windows_dir("APPDATA", ".config")
    return _xdg_dir("XDG_CONFIG_HOME", ".config")


def data_dir() -> Path:
    """Durable application data: the dataset registry, run specs."""
    if (p := _env_path("FLUXKREA_DATA_DIR")) is not None:
        return p
    if (p := _under_home("data")) is not None:
        return p
    if sys.platform == "win32":
        return _windows_dir("LOCALAPPDATA", ".local/share")
    return _xdg_dir("XDG_DATA_HOME", ".local/share")


def cache_dir() -> Path:
    """Rebuildable data: thumbnails, derived metadata, digests."""
    if (p := _env_path("FLUXKREA_CACHE_DIR")) is not None:
        return p
    if (p := _under_home("cache")) is not None:
        return p
    if sys.platform == "win32":
        return _windows_dir("LOCALAPPDATA", ".cache") / "Cache"
    return _xdg_dir("XDG_CACHE_HOME", ".cache")


def state_dir() -> Path:
    """Daemon state that must survive a restart: the job queue, logs."""
    if (p := _env_path("FLUXKREA_STATE_DIR")) is not None:
        return p
    if (p := _under_home("state")) is not None:
        return p
    if sys.platform == "win32":
        return _windows_dir("LOCALAPPDATA", ".local/state") / "State"
    return _xdg_dir("XDG_STATE_HOME", ".local/state")


def _windows_dir(env_var: str, fallback: str) -> Path:
    base = os.environ.get(env_var, "").strip()
    root = expand(base) if base else Path.home().joinpath(*fallback.split("/"))
    return root / APP_DIR_NAME


def _xdg_dir(env_var: str, fallback: str) -> Path:
    base = os.environ.get(env_var, "").strip()
    root = expand(base) if base else Path.home().joinpath(*fallback.split("/"))
    return root / APP_NAME


def config_file() -> Path:
    """The single config file. Precedence is handled in ``core.config``."""
    if (p := _env_path("FLUXKREA_CONFIG_FILE")) is not None:
        return p
    return config_dir() / CONFIG_FILENAME


def fleet_file() -> Path:
    """Client-side node list. See doc 06, "fleet"."""
    if (p := _env_path("FLUXKREA_FLEET_FILE")) is not None:
        return p
    return config_dir() / FLEET_FILENAME


def log_dir() -> Path:
    return state_dir() / "logs"


def queue_dir() -> Path:
    """Persistent job queue, so a daemon restart does not lose work."""
    return state_dir() / "queue"


def runs_dir() -> Path:
    """Generated backend configs and run artifacts, one folder per job."""
    return data_dir() / "runs"


def registry_file() -> Path:
    """Datasets registered with this node (doc 06, ``GET /datasets``)."""
    return data_dir() / "datasets.json"


# --------------------------------------------------------------------------
# bundled assets
# --------------------------------------------------------------------------


def assets_dir() -> Path:
    """Files vendored into the repo - detector weights, and nothing large.

    Vendored rather than downloaded at runtime so the Olympus install
    script needs no extra network fetch (doc 04, "detection").
    """
    if (p := _env_path("FLUXKREA_ASSETS_DIR")) is not None:
        return p
    installed = Path(__file__).resolve().parent.parent / "_assets"
    if installed.is_dir():
        return installed
    # Running from a source checkout: <repo>/assets, three levels up from
    # src/fluxkrea/core/paths.py.
    return Path(__file__).resolve().parents[3] / "assets"


def model_asset(filename: str) -> Path:
    return assets_dir() / "models" / filename


# --------------------------------------------------------------------------
# dataset layout
# --------------------------------------------------------------------------


def masks_dir(dataset_root: str | os.PathLike[str]) -> Path:
    """Sibling folder ai-toolkit reads as ``mask_path`` (doc 04)."""
    return expand(dataset_root) / MASKS_DIRNAME


def preview_dir(dataset_root: str | os.PathLike[str]) -> Path:
    """Redacted previews - a review aid, never consumed by the trainer."""
    return expand(dataset_root) / PREVIEW_DIRNAME


def boxes_file(dataset_root: str | os.PathLike[str]) -> Path:
    """Per-image detected and hand-drawn face boxes (doc 04, "sidecar state")."""
    return expand(dataset_root) / BOXES_FILENAME


def metadata_file(dataset_root: str | os.PathLike[str]) -> Path:
    """Derived metadata cache: quality ratings, review state.

    Never authoritative for caption text - ``.txt`` sidecars are the truth
    (doc 03, "caption storage").
    """
    return expand(dataset_root) / METADATA_FILENAME


def managed_dirs(dataset_root: str | os.PathLike[str]) -> Iterator[Path]:
    """Subfolders the app owns inside a dataset, and the scanner must skip."""
    root = expand(dataset_root)
    yield root / MASKS_DIRNAME
    yield root / PREVIEW_DIRNAME


def describe() -> dict[str, str]:
    """Every resolved location, for ``fk node status`` and ``GET /node``."""
    return {
        "config_dir": str(config_dir()),
        "config_file": str(config_file()),
        "fleet_file": str(fleet_file()),
        "data_dir": str(data_dir()),
        "cache_dir": str(cache_dir()),
        "state_dir": str(state_dir()),
        "log_dir": str(log_dir()),
        "queue_dir": str(queue_dir()),
        "runs_dir": str(runs_dir()),
        "assets_dir": str(assets_dir()),
    }
