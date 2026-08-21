"""Which models exist, and what each one needs. Explicit, not inferred.

v1 decides what it is training by looking for substrings in a path::

    is_flux = 'flux' in model_name.lower() or 'klein' in model_name.lower()
    is_flux2_klein = 'flux.2' in model_name.lower() or 'klein' in model_name.lower()
    is_klein_4b = '4b' in model_name.lower()

So a checkpoint saved as ``my_flux_experiment_4b.safetensors`` trains as
Klein 4B, and ``krea2_raw.safetensors`` in a folder called ``flux`` trains
as something else again. It also emits ``arch: flux2_klein``, which is not
an architecture ai-toolkit has - the real ones are ``flux2_klein_4b`` and
``flux2_klein_9b`` - so those configs fail at load time.

Here each model is named once, with the settings that follow from it.
Nothing is inferred from a filename, and an unknown id raises rather than
falling through to a default (doc 01).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Model:
    """One trainable model, and the facts ai-toolkit needs about it."""

    id: str
    #: ai-toolkit's ``arch`` key. Empty for FLUX.1, which predates arch and
    #: is selected by ``is_flux`` instead.
    arch: str
    label: str
    #: FLUX.1 only. The newer architectures set ``arch`` and leave this false.
    is_flux: bool = False
    #: Sensible LoRA rank. FLUX.2's fused transformer wants more than FLUX.1.
    network_dim: int = 32
    network_alpha: int = 32
    #: Retained only so an existing config that sets it keeps working.
    #: **Not used to decide anything.** ai-toolkit's ``low_vram`` moves the
    #: *transformer* to CPU, and whether that is needed depends on the card
    #: rather than the model - see ``backends/memory.py``.
    low_vram: bool = False
    #: Guidance-distilled models want a higher scale at sample time.
    guidance_scale: float = 4.0
    sample_steps: int = 28
    #: What ai-toolkit loads. A HuggingFace repo id is the portable
    #: default; a node with the weights already on disk overrides it with
    #: ``backends.model_paths`` or ``extra.model_path``, which is a local
    #: ``.safetensors`` file rather than a repo.
    #:
    #: Without this, ``name_or_path`` fell back to the model *id* - the
    #: literal string "flux2" - which is not something anything can load.
    #: Nothing caught it, because a config-shape test only proves the key
    #: is present and `get_job` does not fetch weights.
    repo: str = ""
    #: Transformer size in bf16, gigabytes - measured from the checkpoints,
    #: not from parameter counts. What decides whether this fits on a card.
    weights_gb: float = 0.0
    #: Filenames this model's weights go by on disk, most specific first.
    #: Used to find a checkpoint already sitting in a ComfyUI folder rather
    #: than downloading a second copy of it.
    weight_globs: tuple[str, ...] = field(default_factory=tuple)
    #: Text encoder and VAE the arch pulls in, for ``fk node status`` to
    #: report and for a human to recognise what a run will download.
    text_encoder: str = ""
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_flux2(self) -> bool:
        return self.arch.startswith("flux2")

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "arch": self.arch,
            "label": self.label,
            "repo": self.repo,
            "network_dim": self.network_dim,
            "network_alpha": self.network_alpha,
            "low_vram": self.low_vram,
            "guidance_scale": self.guidance_scale,
            "text_encoder": self.text_encoder,
            "notes": self.notes,
        }


#: Where ComfyUI keeps diffusion weights, newest layout first.
COMFY_SUBDIRS = ("models/diffusion_models", "models/unet", "diffusion_models", "unet")


def find_local_weights(model: Model, roots: Iterable[str | Path]) -> Path | None:
    """A checkpoint for *model* already on this node, or None.

    Carried from v1, including the part that is not obvious: **prefer the
    full-precision file over an fp8 one**. An fp8 checkpoint is for
    inference; training from it starts from a quantised copy of the
    weights, and the LoRA learns the quantisation as well as the subject.
    """
    for root in roots:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        for subdir in COMFY_SUBDIRS:
            folder = base / subdir if (base / subdir).is_dir() else base
            if not folder.is_dir():
                continue
            for pattern in model.weight_globs:
                matches = sorted(
                    folder.glob(pattern),
                    key=lambda path: ("fp8" in path.name.lower(), path.name),
                )
                if matches:
                    return matches[0]
    return None


#: Every model the ai-toolkit backend can train. The arch strings are the
#: ones ai-toolkit actually registers - checked against
#: ``extensions_built_in/diffusion_models/``, not guessed.
MODELS: tuple[Model, ...] = (
    Model(
        id="flux2",
        repo="black-forest-labs/FLUX.2-dev",
        weights_gb=64.0,
        weight_globs=('*flux*2*dev*.safetensors',),
        arch="flux2",
        label="FLUX.2 dev",
        network_dim=32,
        network_alpha=32,
        low_vram=True,
        guidance_scale=4.0,
        text_encoder="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        notes="Guidance-distilled. The 24B Mistral text encoder is the memory cost here.",
        aliases=("flux.2", "flux2-dev", "flux2_dev"),
    ),
    Model(
        id="flux2-klein-4b",
        repo="black-forest-labs/FLUX.2-klein-base-4B",
        weights_gb=7.2,
        weight_globs=('*klein*base*4b*.safetensors', '*klein*4b*.safetensors'),
        arch="flux2_klein_4b",
        label="FLUX.2 Klein 4B",
        network_dim=32,
        network_alpha=32,
        low_vram=False,
        guidance_scale=3.5,
        text_encoder="Qwen/Qwen3-4B",
        notes="Small enough to keep the text encoder on the GPU. Not guidance-distilled.",
        aliases=("klein-4b", "klein4b", "flux2_klein_4b"),
    ),
    Model(
        id="flux2-klein-9b",
        repo="black-forest-labs/FLUX.2-klein-base-9B",
        weights_gb=16.9,
        weight_globs=('*klein*base*9b*.safetensors', '*klein*9b*.safetensors'),
        arch="flux2_klein_9b",
        label="FLUX.2 Klein 9B",
        network_dim=32,
        network_alpha=32,
        low_vram=True,
        guidance_scale=3.5,
        text_encoder="Qwen/Qwen3-8B",
        notes="The text encoder goes to CPU; it does not fit beside the transformer on 16GB.",
        aliases=("klein-9b", "klein9b", "flux2_klein_9b"),
    ),
    Model(
        id="krea2",
        arch="krea2",
        # No public repo: Krea 2 is a local checkpoint here, so a run must
        # be told where it is - backends.model_paths or extra.model_path.
        repo="",
        weights_gb=24.5,
        weight_globs=("*krea2*.safetensors", "*krea*2*.safetensors"),
        label="Krea 2",
        network_dim=32,
        network_alpha=32,
        low_vram=True,
        guidance_scale=4.0,
        text_encoder="Qwen3-VL (resolved by the arch)",
        notes="VAE and text encoder are resolved by the arch; do not set vae_path.",
        aliases=("krea-2", "krea"),
    ),
    Model(
        id="flux1",
        repo="black-forest-labs/FLUX.1-dev",
        weights_gb=23.8,
        weight_globs=('*flux*1*dev*.safetensors', 'flux1*.safetensors'),
        arch="",
        label="FLUX.1 dev",
        is_flux=True,
        network_dim=16,
        network_alpha=8,
        low_vram=False,
        guidance_scale=4.0,
        text_encoder="T5-XXL + CLIP-L",
        notes="Predates ai-toolkit's arch key; selected by is_flux.",
        aliases=("flux", "flux.1", "flux-dev", "flux1-dev"),
    ),
)

_BY_ID: dict[str, Model] = {}
for _model in MODELS:
    _BY_ID[_model.id] = _model
    for _alias in _model.aliases:
        _BY_ID[_alias] = _model


class UnknownModel(Exception):
    """A model id nothing handles. Raised rather than guessed around."""


def normalise(model_id: str) -> str:
    return model_id.strip().lower().replace("_", "-").replace(" ", "-")


def find(model_id: str) -> Model | None:
    """Look up a model by id or alias. Never infers from a path."""
    key = normalise(model_id)
    found = _BY_ID.get(key)
    if found is not None:
        return found
    # Aliases are stored as written; try the normalised spelling of each.
    for candidate, model in _BY_ID.items():
        if normalise(candidate) == key:
            return model
    return None


def get(model_id: str) -> Model:
    model = find(model_id)
    if model is None:
        known = ", ".join(m.id for m in MODELS)
        raise UnknownModel(
            f"unknown model {model_id!r}. Known models: {known}. "
            "v1 inferred the architecture from substrings in the checkpoint path, "
            "which is how a file named for one model trained as another."
        )
    return model


def listing() -> list[dict[str, object]]:
    """Every model, for ``GET /models`` and ``fk models``."""
    return [model.as_dict() for model in MODELS]


__all__ = ["MODELS", "Model", "UnknownModel", "find", "get", "listing", "normalise"]
