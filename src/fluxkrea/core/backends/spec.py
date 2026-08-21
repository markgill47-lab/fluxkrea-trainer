"""``RunSpec`` - one typed description of a training run.

One source of truth, N renderers. Each backend turns this into its own
config format, instead of v1's independent ``generate_config`` methods
drifting apart (doc 02). It is also the payload of ``POST /jobs``.

It lives in ``core`` rather than in the daemon because the backend
protocol takes one, and ``core`` may not reach up into its own clients.
The daemon imports it; it imports nothing of the daemon's.

Deliberately open at the edges: ``extra`` carries backend-specific
settings that have not earned a field, so a backend can pass something
through without a schema change every time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class RunSpec:
    """What to train, on what, with which knobs."""

    model: str
    dataset: str
    name: str = ""
    output: str = ""
    #: Which GPU to pin to. One queue slot per device.
    device: int = 0
    steps: int = 0
    batch_size: int = 1
    learning_rate: float = 0.0
    network_dim: int = 0
    network_alpha: int = 0
    resolution: int = 0
    #: The mask folder, when the run is masked. The one line doc 04 exists
    #: to produce, threaded from the dataset through to the trainer config.
    mask_path: str = ""
    #: ai-toolkit's ``mask_min_value``. 0.0 means the region is fully ignored.
    mask_min_value: float = 0.0
    sample_every: int = 0
    save_every: int = 0
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def masked(self) -> bool:
        return bool(self.mask_path)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunSpec:
        """Build from a request body, keeping what the schema does not name."""
        known = {f.name for f in fields(cls)}
        kept = {k: v for k, v in data.items() if k in known}
        unknown = {k: v for k, v in data.items() if k not in known}

        if not kept.get("model"):
            raise ValueError("a run spec needs a model")
        if not kept.get("dataset"):
            raise ValueError("a run spec needs a dataset")

        spec = cls(**kept)
        if unknown:
            spec.extra.update(unknown)
        return spec


__all__ = ["RunSpec"]
