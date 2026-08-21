"""One typed config, one file, explicit precedence.

v1 has four overlapping stores with no documented precedence, two of them
gitignored because they hold API keys - which means the application's real
configuration cannot be shared between the Windows desk and the Linux
fleet (doc 01, "config sprawl").

v2 has one file and one schema::

    defaults in code  <  config file  <  environment  <  CLI flags

and **no secrets in it at all**. API keys come from the environment or an
OS keyring (see :func:`secret`), which is what makes the config file
committable and shareable across the fleet. A secret-looking key in the
file is a hard error, not a warning - the whole point is that the file
stays safe to check in.

Environment keys are ``FLUXKREA_<SECTION>_<FIELD>``, upper-cased::

    FLUXKREA_MASK_EXPAND=1.8
    FLUXKREA_DAEMON_PORT=8472
    FLUXKREA_DATASET_ROOTS=D:/LoRA_Training_data:/srv/datasets

CLI flags arrive as a flat mapping of dotted keys, ``{"mask.expand": 1.8}``,
which is also the shape the API's settings endpoint will speak.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
import types
import typing
from collections.abc import Iterator
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from . import paths

# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

#: One extension list for the whole application. v1 keeps two copies in two
#: classes, which is how its gallery and its processor come to disagree
#: about what exists in a folder (doc 03).
DEFAULT_IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".avif",
    ".jfif",
]


@dataclass(slots=True)
class DatasetConfig:
    """Dataset discovery and the rules a dataset is validated against."""

    #: Folders the daemon will browse and register datasets under. Empty
    #: means "no restriction", which is only safe on a localhost bind.
    roots: list[Path] = field(default_factory=list)
    image_extensions: list[str] = field(default_factory=lambda: list(DEFAULT_IMAGE_EXTENSIONS))
    #: ``caption_ext: txt`` is what every trainer config in use reads.
    caption_ext: str = "txt"
    #: ``validate`` flags images below this on the short edge.
    min_resolution: int = 512
    #: Follow subfolders when scanning. Off by default: a dataset folder is
    #: flat, and ``masks/`` and ``preview/`` are ours, not training data.
    recursive: bool = False


@dataclass(slots=True)
class MaskConfig:
    """Face masking (doc 04). Defaults chosen for recall, not precision.

    A false positive costs a wasted region; a false negative puts an
    unmasked face into training and defeats the feature.
    """

    detector: str = "yunet"
    #: Deliberately low. Missed faces are the expensive failure.
    confidence: float = 0.5
    nms: float = 0.3
    #: Detectors return an eyes-to-chin box; identity lives in hair, hairline
    #: and jaw as well.
    expand: float = 1.6
    #: Extra growth applied upward only, to catch the hairline.
    expand_up: float = 1.35
    #: Pixels of deliberate gradient at the mask boundary. Feathering is
    #: applied here, never acquired accidentally through a resize (doc 03).
    feather: int = 6
    #: ai-toolkit ``mask_min_value``. 0.0 means the region is fully ignored.
    min_value: float = 0.0
    #: Faces black, everything else white. Flip only if the trainer's
    #: polarity ever changes.
    invert: bool = False
    #: Export refuses to run on unreviewed or zero-detection images unless
    #: explicitly overridden.
    require_review: bool = True
    #: Also write redacted previews. A review aid; the trainer reads masks.
    write_previews: bool = True


@dataclass(slots=True)
class CaptionerConfig:
    """Captioning. The API key is *not* here - see :func:`secret`."""

    provider: str = "claude"
    model: str = "claude-sonnet-5"
    ollama_url: str = "http://localhost:11434"
    max_concurrent: int = 4


@dataclass(slots=True)
class DaemonConfig:
    """The per-node HTTP daemon (doc 06)."""

    #: Localhost by default. The API launches processes and rewrites dataset
    #: folders; remote access is an SSH tunnel, not an open port.
    host: str = "127.0.0.1"
    port: int = 8471
    #: Name this node reports as. Empty means the machine hostname.
    node_name: str = ""
    #: Worker threads for dataset tasks. Training jobs get their own slots.
    workers: int = 2


@dataclass(slots=True)
class BackendsConfig:
    """Where the training stacks live on this node."""

    #: Checkout of ai-toolkit that serves both FLUX and Krea 2.
    aitoolkit_path: Path | None = None
    #: Interpreter to launch training with. Empty means the current one.
    python_exe: str = ""
    #: Where generated configs and checkpoints land. Empty means ``runs_dir``.
    output_root: Path | None = None


@dataclass(slots=True)
class Config:
    """The whole configuration. Load with :func:`load`."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    captioner: CaptionerConfig = field(default_factory=CaptionerConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    backends: BackendsConfig = field(default_factory=BackendsConfig)
    log_level: str = "info"

    #: Where this instance was loaded from. Not written back out.
    source: Path | None = field(default=None, compare=False)

    # -- serialisation ----------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """Plain nested dict, JSON- and TOML-ready. Never contains secrets."""
        return {
            name: _plain(getattr(self, name))
            for name in _section_names()
            if name != "source"
        }

    def get(self, dotted: str) -> Any:
        """Read one value by dotted key, as the API and CLI address it."""
        section, _, name = dotted.partition(".")
        if not name:
            return getattr(self, section)
        return getattr(getattr(self, section), name)

    def set(self, dotted: str, value: Any) -> None:
        """Write one value by dotted key, coercing to the declared type."""
        section, _, name = dotted.partition(".")
        if not name:
            raise KeyError(f"{dotted!r} is a section, not a setting")
        target = getattr(self, section, None)
        if target is None or not is_dataclass(target):
            raise KeyError(f"unknown config section {section!r}")
        declared = {f.name: f.type for f in fields(target)}
        if name not in declared:
            raise KeyError(f"unknown setting {dotted!r}")
        setattr(target, name, _coerce(value, declared[name], dotted))

    def save(self, path: Path | None = None) -> Path:
        """Write the config file. Secrets are never included, by construction."""
        import tomli_w

        target = path or paths.config_file()
        paths.ensure_dir(target.parent)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(tomli_w.dumps(_toml_ready(self.as_dict())).encode("utf-8"))
        tmp.replace(target)
        return target

    # -- validation -------------------------------------------------------

    def validate(self) -> list[str]:
        """Problems worth refusing to start over. Returns messages, raises nothing."""
        problems: list[str] = []

        if not (0 < self.daemon.port < 65536):
            problems.append(f"daemon.port {self.daemon.port} is out of range")

        # Doc 06, security: binding beyond loopback without a token would put
        # process launching and dataset rewriting on the network.
        if self.daemon.host not in ("127.0.0.1", "localhost", "::1") and not secret("token"):
            problems.append(
                f"daemon.host is {self.daemon.host!r} but no FLUXKREA_TOKEN is set; "
                "the daemon refuses to listen beyond localhost without one"
            )

        if self.mask.expand < 1.0:
            problems.append("mask.expand below 1.0 would shrink the detected box")
        if not 0.0 <= self.mask.min_value <= 1.0:
            problems.append("mask.min_value must be between 0.0 and 1.0")
        if self.mask.feather < 0:
            problems.append("mask.feather cannot be negative")
        if not 0.0 < self.mask.confidence <= 1.0:
            problems.append("mask.confidence must be between 0.0 and 1.0")

        bad_ext = [e for e in self.dataset.image_extensions if not e.startswith(".")]
        if bad_ext:
            problems.append(f"dataset.image_extensions must start with a dot: {bad_ext}")

        for root in self.dataset.roots:
            if not root.exists():
                problems.append(f"dataset root does not exist: {root}")

        if self.backends.aitoolkit_path and not self.backends.aitoolkit_path.exists():
            problems.append(f"backends.aitoolkit_path does not exist: {self.backends.aitoolkit_path}")

        return problems


# --------------------------------------------------------------------------
# secrets
# --------------------------------------------------------------------------

#: Anything matching these never belongs in the config file.
SECRET_HINTS = ("key", "token", "secret", "password", "passwd", "credential")

#: Logical secret name -> environment variables to try, in order. The
#: aliases exist so an existing v1 shell profile keeps working.
SECRET_ENV: dict[str, tuple[str, ...]] = {
    "claude": ("FLUXKREA_CLAUDE_API_KEY", "ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
    "token": ("FLUXKREA_TOKEN",),
    "hf": ("FLUXKREA_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
}


def secret(name: str) -> str | None:
    """Fetch a secret from the environment, falling back to the OS keyring.

    Never from the config file. This is the single rule that lets the config
    be committed and shared across the fleet (doc 06, "security").
    """
    for env_var in SECRET_ENV.get(name, (f"FLUXKREA_{name.upper()}",)):
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    try:  # optional dependency; absent on a bare fleet node
        import keyring
    except Exception:  # noqa: BLE001
        return None
    try:
        return keyring.get_password(paths.APP_NAME, name) or None
    except Exception:  # noqa: BLE001 - a locked or missing backend is not fatal
        return None


class ConfigError(Exception):
    """A config file that cannot be honoured. Never guessed around."""


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load(
    path: Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    use_file: bool = True,
) -> Config:
    """Build a :class:`Config` with the documented precedence.

    ``defaults in code < config file < environment < CLI flags``

    *overrides* is a flat mapping of dotted keys - what a CLI parser or the
    settings endpoint produces. ``None`` values in it are ignored, so an
    unset flag does not clobber a configured value.
    """
    cfg = Config()

    if use_file:
        target = path or paths.config_file()
        if target.is_file():
            _apply_file(cfg, target)
            cfg.source = target
        elif path is not None:
            raise ConfigError(f"config file not found: {target}")

    _apply_env(cfg, os.environ if env is None else env)

    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        cfg.set(dotted, value)

    return cfg


def _apply_file(cfg: Config, target: Path) -> None:
    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {target}: {exc}") from exc

    for section, values in data.items():
        if section not in _section_names():
            raise ConfigError(f"{target}: unknown section [{section}]")
        if not isinstance(values, dict):
            # A top-level scalar such as log_level.
            setattr(cfg, section, _coerce(values, _top_level_types()[section], section))
            continue
        for name, value in values.items():
            dotted = f"{section}.{name}"
            if _looks_secret(name):
                raise ConfigError(
                    f"{target}: {dotted!r} looks like a secret. "
                    "Secrets come from the environment or the OS keyring so that "
                    "this file stays safe to commit and share across the fleet."
                )
            try:
                cfg.set(dotted, value)
            except KeyError as exc:
                raise ConfigError(f"{target}: {exc}") from exc


def _apply_env(cfg: Config, env: dict[str, str] | os._Environ[str]) -> None:
    for dotted, declared in _all_settings():
        section, _, name = dotted.partition(".")
        key = f"FLUXKREA_{section.upper()}_{name.upper()}"
        raw = env.get(key)
        if raw is None or not raw.strip():
            continue
        cfg.set(dotted, _from_env_string(raw, declared, dotted))

    top = env.get("FLUXKREA_LOG_LEVEL")
    if top and top.strip():
        cfg.log_level = top.strip()


def _looks_secret(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in SECRET_HINTS)


# --------------------------------------------------------------------------
# type plumbing
# --------------------------------------------------------------------------


def _section_names() -> tuple[str, ...]:
    return tuple(f.name for f in fields(Config))


def _top_level_types() -> dict[str, Any]:
    return {f.name: f.type for f in fields(Config)}


def _all_settings() -> Iterator[tuple[str, Any]]:
    """Every dotted key in the schema, with its declared type."""
    for section in fields(Config):
        default = getattr(Config(), section.name, None)
        if not is_dataclass(default):
            continue
        for setting in fields(default):
            yield f"{section.name}.{setting.name}", setting.type


def _from_env_string(raw: str, declared: Any, dotted: str) -> Any:
    """Environment values are always strings, and lists arrive as a path list."""
    raw = raw.strip()
    origin, args = _unwrap(declared)
    if origin is list:
        # os.pathsep so a Windows value can carry drive letters unambiguously,
        # with a comma accepted for the extension lists where it reads better.
        parts = raw.split(os.pathsep) if os.pathsep in raw else raw.split(",")
        return [_coerce(p.strip(), args[0] if args else str, dotted) for p in parts if p.strip()]
    return _coerce(raw, declared, dotted)


def _unwrap(declared: Any) -> tuple[Any, tuple[Any, ...]]:
    """Resolve a possibly-stringified annotation to ``(origin, args)``.

    ``from __future__ import annotations`` means every field type reaches us
    as a string, so the common shapes are matched textually rather than
    dragging in a full ``typing.get_type_hints`` resolution.
    """
    if isinstance(declared, str):
        text = declared.replace(" ", "")
        text = text.removesuffix("|None")
        if text.startswith("list["):
            inner = text[len("list[") : -1]
            return list, ({"Path": Path, "str": str, "int": int, "float": float}.get(inner, str),)
        return {"Path": Path, "str": str, "int": int, "float": float, "bool": bool}.get(text, text), ()
    origin = typing.get_origin(declared)
    if origin in (typing.Union, types.UnionType):
        args = tuple(a for a in typing.get_args(declared) if a is not type(None))
        return _unwrap(args[0]) if len(args) == 1 else (origin, args)
    return (origin or declared), typing.get_args(declared)


def _optional(declared: Any) -> bool:
    return "None" in declared if isinstance(declared, str) else type(None) in typing.get_args(declared)


def _coerce(value: Any, declared: Any, dotted: str) -> Any:
    """Coerce a TOML / env / flag value to the field's declared type."""
    if value is None:
        if _optional(declared):
            return None
        raise ConfigError(f"{dotted} cannot be null")

    target, args = _unwrap(declared)

    if target is list:
        inner = args[0] if args else str
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{dotted} expects a list, got {type(value).__name__}")
        return [_coerce(v, inner, dotted) for v in value]

    if target is Path:
        text = str(value).strip()
        return paths.expand(text) if text else None if _optional(declared) else Path(text)

    if target is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"{dotted} expects a boolean, got {value!r}")

    if target is int:
        try:
            return int(str(value).strip())
        except ValueError as exc:
            raise ConfigError(f"{dotted} expects an integer, got {value!r}") from exc

    if target is float:
        try:
            return float(str(value).strip())
        except ValueError as exc:
            raise ConfigError(f"{dotted} expects a number, got {value!r}") from exc

    if target is str:
        return str(value)

    return value


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Path):
        # Forward slashes: the same file is read on Windows and Linux.
        return value.as_posix()
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def _toml_ready(data: dict[str, Any]) -> dict[str, Any]:
    """TOML has no null. Drop unset optionals rather than inventing a value."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, dict):
            nested = _toml_ready(value)
            out[key] = nested
        else:
            out[key] = value
    # TOML requires scalars before tables; dataclass order already gives us
    # sections first, so hoist the bare scalars to the front.
    scalars = {k: v for k, v in out.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in out.items() if isinstance(v, dict)}
    return {**scalars, **tables}


def example_toml() -> str:
    """A commented starter file, for ``fk config init``."""
    return (
        "# FluxKrea Trainer configuration.\n"
        "# Precedence: defaults in code < this file < environment < CLI flags.\n"
        "# Secrets never live here - export FLUXKREA_CLAUDE_API_KEY instead.\n"
        "# Every setting is also FLUXKREA_<SECTION>_<FIELD> in the environment.\n\n"
        + __import__("tomli_w").dumps(_toml_ready(Config().as_dict()))
    )


def dataclass_sections() -> tuple[type, ...]:
    """The section types, for tests and for documentation generation."""
    return tuple(
        type(getattr(Config(), f.name))
        for f in dataclasses.fields(Config)
        if is_dataclass(getattr(Config(), f.name, None))
    )
