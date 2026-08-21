"""Ollama — a vision model running locally.

The one captioner that needs no API key and sends no image anywhere. For a
lab that trains on reference photography that matters more than it looks:
the alternative is uploading every frame of a dataset to somebody else's
service to have it described.

Ported from v1's ``captioners/ollama_backend.py``, keeping its two good
decisions — probe the daemon *and* the model separately, and tolerate a
model named without its ``:latest`` tag — and adding a third: report which
models *are* installed when the wanted one is not, because "model not
found" without that list is a guessing game.
"""

from __future__ import annotations

from pathlib import Path

from .base import DEFAULT_PROMPT, Captioner, encode_image, get_json, post_json

DEFAULT_URL = "http://localhost:11434"

#: A vision model that is small enough to run beside a training job and
#: uncensored enough not to refuse ordinary reference photography, which
#: general-purpose models do often enough to be a problem for this use.
DEFAULT_MODEL = "llama3.2-vision"

#: Vision models are slow on CPU and not fast on a busy GPU. This is a
#: ceiling for one image, not an expectation.
DEFAULT_TIMEOUT = 180.0


class OllamaCaptioner(Captioner):
    name = "ollama"
    label = "Ollama (local)"

    def __init__(
        self,
        url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        temperature: float = 0.4,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    # -- probing -----------------------------------------------------------

    def installed_models(self) -> tuple[bool, list[str] | str]:
        ok, body = get_json(f"{self.url}/api/tags")
        if not ok:
            return False, str(body)
        try:
            return True, sorted(entry["name"] for entry in body.get("models", []))
        except (AttributeError, KeyError, TypeError):
            return False, "Ollama returned an unexpected model list"

    def test(self) -> tuple[bool, str]:
        """Two separate checks, because they fail for different reasons.

        A daemon that is not running and a model that is not pulled need
        different fixes, and one message covering both helps nobody.
        """
        ok, models = self.installed_models()
        if not ok:
            return False, (
                f"Cannot reach Ollama at {self.url}. Is `ollama serve` running? ({models})"
            )

        assert isinstance(models, list)
        # `models[].name` is e.g. "llava:latest"; tolerate the tag being
        # omitted in configuration, which is how people usually write it.
        wanted = self.model.split(":")[0]
        if not any(name.split(":")[0] == wanted for name in models):
            listed = ", ".join(models) or "(none installed)"
            return False, (
                f"Model '{self.model}' is not pulled. Run: ollama pull {self.model}\n"
                f"Installed: {listed}"
            )
        return True, f"Ollama at {self.url}, model '{self.model}' ready."

    # -- captioning --------------------------------------------------------

    def describe(
        self,
        image: Path,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 400,
    ) -> tuple[bool, str]:
        if not image.is_file():
            return False, f"image not found: {image}"

        ok, body = post_json(
            f"{self.url}/api/generate",
            {
                "model": self.model,
                "prompt": prompt or DEFAULT_PROMPT,
                "images": [encode_image(image)],
                "stream": False,
                "options": {
                    "num_predict": int(max_tokens),
                    # Above the default for a more descriptive caption, but
                    # not so high that it starts inventing detail.
                    "temperature": self.temperature,
                },
            },
            timeout=self.timeout,
        )
        if not ok:
            return False, f"Ollama: {body}"

        text = str(body.get("response", "")).strip()
        if not text:
            return False, "Ollama returned an empty response"
        return True, text
