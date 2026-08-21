"""Choose memory settings from the card, not from a constant.

The bug this exists for: ``low_vram`` was a per-model constant in the
registry, set for Klein 9B because "it does not fit beside the transformer
on 16GB". On a 31.8GB card that is simply false — and what the flag
actually does is not what the comment said:

    flux2_model.py, load_model()
    if self.model_config.low_vram:
        self.print_and_status_update("Moving transformer to CPU")
        transformer.to("cpu")

It moves the **transformer** to CPU. So a 9B model that fits in VRAM twice
over was streamed across PCIe instead, 12GB of it living in Windows'
shared memory, and an hour-long run became an eleven-hour one at 98% GPU
utilisation — busy, but busy waiting.

None of these settings are properties of a model. They are answers to
"does this model fit on this card", which needs both halves. A fleet with
16GB and 32GB cards in it cannot have one right answer baked into a table.

The numbers below are deliberately coarse. Being roughly right about
whether 17GB of weights fit in 32GB is the whole job; being precise about
activation memory is not achievable from here and not needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Training needs the weights plus activations, gradients for the LoRA and
#: optimiser state. With gradient checkpointing on - which every config
#: here sets - this multiple has held for FLUX-family LoRA training.
BF16_HEADROOM = 1.30

#: What quanto's int8 leaves behind. Not 0.5: normalisations, embeddings
#: and the layers it declines to quantise stay wide.
INT8_FACTOR = 0.55

#: Quantised training carries more transient state, so a little more room.
INT8_HEADROOM = 1.35


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    """How to fit a model on a card, and why it says so."""

    quantize: bool
    quantize_te: bool
    low_vram: bool
    layer_offloading: bool
    #: Only meaningful when ``layer_offloading``. 1.0 is the whole model.
    transformer_percent: float
    reason: str

    def as_config(self) -> dict[str, Any]:
        """The keys ai-toolkit reads. Offload percentages only when on."""
        block: dict[str, Any] = {
            "quantize": self.quantize,
            "quantize_te": self.quantize_te,
            "low_vram": self.low_vram,
            "layer_offloading": self.layer_offloading,
        }
        if self.layer_offloading:
            block["layer_offloading_transformer_percent"] = self.transformer_percent
            block["layer_offloading_text_encoder_percent"] = 1.0
        return block


def plan_memory(weights_gb: float, vram_gb: float | None) -> MemoryPlan:
    """Pick settings for *weights_gb* of model on a card of *vram_gb*.

    With no card information, returns the cautious plan rather than
    guessing generously: a run that is slower than it needed to be is a
    worse outcome than one that will not start, but only just, and the
    caller can always override.
    """
    if not weights_gb or not vram_gb:
        return MemoryPlan(
            quantize=True,
            quantize_te=True,
            low_vram=False,
            layer_offloading=False,
            transformer_percent=1.0,
            reason="no GPU detected, so these are cautious defaults",
        )

    if vram_gb >= weights_gb * BF16_HEADROOM:
        return MemoryPlan(
            quantize=False,
            quantize_te=False,
            low_vram=False,
            layer_offloading=False,
            transformer_percent=1.0,
            reason=(
                f"{weights_gb:.1f}GB of weights fit in {vram_gb:.1f}GB at full "
                "precision, so nothing is quantised or offloaded"
            ),
        )

    quantized = weights_gb * INT8_FACTOR
    if vram_gb >= quantized * INT8_HEADROOM:
        return MemoryPlan(
            quantize=True,
            quantize_te=True,
            low_vram=False,
            layer_offloading=False,
            transformer_percent=1.0,
            reason=(
                f"{weights_gb:.1f}GB does not fit in {vram_gb:.1f}GB at full "
                f"precision; quantised to about {quantized:.1f}GB it does"
            ),
        )

    # Past this the card cannot hold the model however it is compressed, so
    # part of it lives in system memory and the run pays PCIe for it. Offload
    # only the excess: the difference between a slow run and no run.
    room = max(vram_gb / (quantized * INT8_HEADROOM), 0.0)
    percent = min(max(1.0 - room, 0.1), 1.0)
    return MemoryPlan(
        quantize=True,
        quantize_te=True,
        low_vram=False,
        layer_offloading=True,
        transformer_percent=round(percent, 2),
        reason=(
            f"{weights_gb:.1f}GB will not fit in {vram_gb:.1f}GB even quantised; "
            f"offloading {percent:.0%} of the transformer, which will be slow"
        ),
    )


def detect_vram_gb(device: int = 0) -> float | None:
    """Total VRAM on *device*, or None when there is no CUDA card here.

    A lazy import: this package has no torch of its own on a laptop, and
    asking about a GPU must not be the thing that makes it need one.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available() or device >= torch.cuda.device_count():
            return None
        _, total = torch.cuda.mem_get_info(device)
    except Exception:  # noqa: BLE001 - a driver that will not answer is a None
        return None
    return total / (1024**3)


__all__ = ["BF16_HEADROOM", "MemoryPlan", "detect_vram_gb", "plan_memory"]
