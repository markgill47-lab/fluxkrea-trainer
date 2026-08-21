"""JoyCaption — a LLaVA-format model loaded in this process.

The captioner this project was actually built around. It is not an Ollama
model and never appears in ``ollama list``: it is a HuggingFace repo
loaded through ``transformers``, cached under ``~/.cache/huggingface``.
The two are unrelated systems, which is worth saying plainly because the
symptom of confusing them is a model you know you downloaded not being
listed anywhere.

Why it is worth the 16GB and the VRAM: it was fine-tuned for exactly this
job — writing training captions — and it does not refuse art, anatomy,
named subjects or NSFW. General-purpose vision models refuse ordinary
reference photography often enough to make a batch unusable, and a
refusal costs more than a bad caption because it leaves the image
uncaptioned.

Ported from v1's ``captioners/joycaption_backend.py`` with its inference
path intact, including the chat-template detail that took a while to find
(see :meth:`JoyCaptionCaptioner.describe`).

**VRAM, on an 8B Llama base plus SigLIP:**

===========  ========
bf16         ~17 GB
int8 quanto  ~9 GB
int4 quanto  ~5 GB
===========  ========

Quantized by default. A 32GB card holding a training job as well does not
have 17GB spare, and :meth:`close` exists to hand the rest back — the
batch operation calls it when the run ends.
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Any

from .base import DEFAULT_PROMPT, Captioner, CaptionerError

DEFAULT_MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"

#: JoyCaption was trained with a chat-style system prompt. This is the
#: generic one; the caller's prompt becomes the user turn.
SYSTEM_PROMPT = (
    "You are a helpful image captioner. Describe the image in detail. "
    "Be objective and precise. Do not refuse based on subject matter."
)


class JoyCaptionCaptioner(Captioner):
    name = "joycaption"
    label = "JoyCaption (local, in-process)"

    def __init__(
        self,
        model: str = DEFAULT_MODEL_ID,
        quantize: bool = True,
        device: str | None = None,
        dtype: str = "bfloat16",
        temperature: float = 0.6,
    ) -> None:
        self.model_id = model
        self.quantize = quantize
        self.device = device
        self.dtype = dtype
        self.temperature = temperature
        self._processor: Any = None
        self._model: Any = None
        # Loading is slow and not reentrant; two captions arriving together
        # must not both materialise 17GB of weights.
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        """Materialise the weights. Safe to call repeatedly."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            try:
                import torch
                from transformers import AutoProcessor, LlavaForConditionalGeneration
            except ImportError as exc:
                raise CaptionerError(
                    "JoyCaption needs torch and transformers: "
                    'pip install "fluxkrea[joycaption]"'
                ) from exc

            device = self._resolve_device()
            torch_dtype = getattr(torch, self.dtype, torch.bfloat16)

            processor = AutoProcessor.from_pretrained(self.model_id)
            model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map="cpu",  # staged on CPU; moved and quantized below
            )

            if self.quantize:
                try:
                    from optimum.quanto import freeze, qint8
                    from optimum.quanto import quantize as quanto_quantize
                except ImportError:
                    # Not fatal: it still runs, in twice the VRAM. Saying so
                    # beats failing a batch over a missing optional package.
                    pass
                else:
                    # Only the language head. The vision tower's weights are
                    # small next to it and quantizing them buys little.
                    target = getattr(model, "language_model", model)
                    quanto_quantize(target, weights=qint8)
                    freeze(target)

            self._model = model.to(device)
            self._model.eval()
            self._processor = processor

    def close(self) -> None:
        """Hand the VRAM back.

        Not optional housekeeping on this node: the same GPU runs training,
        and 17GB still held by a finished caption run is 17GB the next job
        does not get.
        """
        if self._model is None:
            return
        with self._lock:
            self._model = None
            self._processor = None
            gc.collect()
            try:
                import torch
            except ImportError:
                return
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # -- probing -----------------------------------------------------------

    def test(self) -> tuple[bool, str]:
        """Answer without materialising 16GB of weights.

        The useful question before a batch is "will this start", and the
        expensive part of starting is the download. A cache probe answers
        that in milliseconds; loading to find out would take minutes and
        fill the card.
        """
        if self._model is not None:
            return True, f"JoyCaption loaded: {self.model_id}"

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return False, (
                "JoyCaption needs torch, transformers and huggingface_hub: "
                'pip install "fluxkrea[joycaption]"'
            )

        try:
            import transformers  # noqa: F401
        except ImportError:
            return False, (
                'JoyCaption needs transformers: pip install "fluxkrea[joycaption]"'
            )

        try:
            # Succeeds if and only if the snapshot is already cached.
            snapshot_download(self.model_id, local_files_only=True)
        except Exception:  # noqa: BLE001 - hub raises several unrelated types
            return False, (
                f"'{self.model_id}' is not in the local HuggingFace cache. "
                "It is about 16GB and will download on first use. "
                "Fetch it now with: "
                f"huggingface-cli download {self.model_id}"
            )

        where = "cuda" if self._resolve_device() == "cuda" else "cpu (slow)"
        vram = "int8" if self.quantize else "bf16"
        return True, (
            f"JoyCaption cached and ready: {self.model_id} on {where}, {vram}. "
            "Loads on the first caption."
        )

    # -- captioning --------------------------------------------------------

    def describe(
        self,
        image: Path,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 400,
    ) -> tuple[bool, str]:
        if not image.is_file():
            return False, f"image not found: {image}"

        try:
            self._load()
        except CaptionerError as exc:
            return False, str(exc)
        except Exception as exc:  # noqa: BLE001 - a load failure is one image's problem
            return False, f"cannot load JoyCaption: {type(exc).__name__}: {exc}"

        try:
            import torch
            from PIL import Image as PILImage

            with PILImage.open(image) as opened:
                picture = opened.convert("RGB")

            # JoyCaption uses a Llama-3.1 style chat template where `content`
            # is a plain string. The image goes to the processor separately
            # via `images=`, NOT as a typed content block - a list of dicts
            # here breaks Jinja with "'list object' has no attribute
            # 'replace'". This cost an afternoon in v1; it stays written down.
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt or DEFAULT_PROMPT},
            ]
            templated = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = self._processor(text=[templated], images=[picture], return_tensors="pt")
            inputs = {
                key: value.to(self._model.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=int(max_tokens),
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    suppress_tokens=None,
                )

            # Everything before this is the prompt echoed back.
            prompt_length = inputs["input_ids"].shape[-1]
            text = self._processor.tokenizer.decode(
                generated[0, prompt_length:], skip_special_tokens=True
            ).strip()
        except Exception as exc:  # noqa: BLE001 - one bad image must not end the batch
            return False, f"JoyCaption inference error: {type(exc).__name__}: {exc}"

        if not text:
            return False, "JoyCaption returned an empty response"
        return True, text
