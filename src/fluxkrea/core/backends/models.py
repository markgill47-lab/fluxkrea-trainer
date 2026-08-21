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

from dataclasses import dataclass, field


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
    #: Keep the text encoder off the GPU. The 9B encoder does not fit
    #: alongside the transformer on a 16GB card; the 4B one does.
    low_vram: bool = False
    #: Guidance-distilled models want a higher scale at sample time.
    guidance_scale: float = 4.0
    sample_steps: int = 28
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
            "network_dim": self.network_dim,
            "network_alpha": self.network_alpha,
            "low_vram": self.low_vram,
            "guidance_scale": self.guidance_scale,
            "text_encoder": self.text_encoder,
            "notes": self.notes,
        }


#: Every model the ai-toolkit backend can train. The arch strings are the
#: ones ai-toolkit actually registers - checked against
#: ``extensions_built_in/diffusion_models/``, not guessed.
MODELS: tuple[Model, ...] = (
    Model(
        id="flux2",
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
