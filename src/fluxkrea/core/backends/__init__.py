"""The training backend interface, and the registry that dispatches to it.

v1 has three stacks implementing the same lifecycle, each independently
defining ``start_training``, ``stop_training``, ``is_training_running``,
``get_progress`` and ``get_loss_history``. An interface already exists
there - it is just never written down, never type checked, and never
uniformly honoured, so the features are badly asymmetric: Klein alone has
trend detection, outliers, EMA, metric export and live config updates,
and anything built on the richer API silently degrades on the other
backends (doc 01).

Here it is written down once. Two implementations survive the cut -
ai-toolkit (FLUX and Krea 2 collapse into one config-driven class) and
Klein - and both arrive in P4/P5. This module is the socket they plug
into.

**Notably absent from the protocol:** ``get_loss_history``, trend
detection, outliers, EMA. Those are not backend concerns. Backends emit
``LossPoint`` events; ``analytics/loss.py`` consumes the stream and
computes the rest for every backend equally.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..events import Emitter
from .spec import RunSpec


@dataclass(frozen=True, slots=True)
class BackendProgress:
    """Where a run has got to, as the backend understands it."""

    step: int = 0
    total: int = 0
    epoch: int = 0
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "total": self.total,
            "epoch": self.epoch,
            "message": self.message,
        }


@runtime_checkable
class TrainingBackend(Protocol):
    """The lifecycle v1 has three unwritten copies of."""

    name: str

    def supports(self, model_id: str) -> bool:
        """True if this backend can train *model_id*."""
        ...

    def generate_config(self, run: RunSpec) -> Path:
        """Render the spec into this backend's own config format.

        Generated configs are *artifacts*, written from the spec, never
        hand-edited as the source of truth.
        """
        ...

    def start(self, config_path: Path, emit: Emitter, cancel: threading.Event) -> None:
        """Run the training. Blocking; the queue calls this on a worker thread."""
        ...

    def stop(self) -> None: ...

    def is_running(self) -> bool: ...

    def progress(self) -> BackendProgress: ...


class BackendError(Exception):
    """A backend that cannot run, or a model nothing handles."""


_REGISTRY: dict[str, TrainingBackend] = {}


def register(backend: TrainingBackend) -> TrainingBackend:
    """Add a backend. Called at import time by each implementation."""
    _REGISTRY[backend.name] = backend
    return backend


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def registered() -> dict[str, TrainingBackend]:
    return dict(_REGISTRY)


def get(name: str) -> TrainingBackend:
    backend = _REGISTRY.get(name)
    if backend is None:
        known = ", ".join(sorted(_REGISTRY)) or "none registered"
        raise BackendError(f"unknown backend {name!r}; available: {known}")
    return backend


def supported_by(model_id: str) -> TrainingBackend | None:
    """Which backend handles *model_id*, or ``None``.

    **Explicit dispatch, and no fallback.** v1's ``detect_backend`` ends
    with ``return 'kohya'``, so an unrecognised model silently routes to a
    trainer that cannot handle it. Returning ``None`` here is what lets the
    caller raise instead.
    """
    for backend in _REGISTRY.values():
        try:
            if backend.supports(model_id):
                return backend
        except Exception:  # noqa: BLE001 - a broken backend must not hide the others
            continue
    return None


__all__ = [
    "BackendError",
    "BackendProgress",
    "RunSpec",
    "TrainingBackend",
    "get",
    "register",
    "registered",
    "supported_by",
    "unregister",
]
