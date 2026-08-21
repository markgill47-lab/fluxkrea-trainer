"""Captioners, behind an interface.

Doc 01 singles this part of v1 out as already correctly shaped — "a clean
``Captioner`` base with Claude, Ollama and JoyCaption backends behind a
factory" — so the shape survives the rewrite. What changes is what sits
behind it: no ``requests``, and failures returned instead of raised (see
:mod:`~fluxkrea.core.captioners.base`).

Two of the three are local. ``ollama`` is the default because it needs
nothing installed into this process; ``joycaption`` is the one this
project was built around — a LLaVA model loaded in-process, fine-tuned
for training captions and unwilling to refuse ordinary reference
photography. ``claude`` writes the best captions and is the only one that
sends an image anywhere.
"""

from __future__ import annotations

from typing import Any

from .base import DEFAULT_PROMPT, Captioner, CaptionerError

#: What ``caption.backend`` defaults to in the config.
DEFAULT_BACKEND = "ollama"


def get_captioner(name: str, **options: Any) -> Captioner:
    """Build a captioner by name. The config's ``caption.backend`` comes here.

    Imports are deferred per backend so that a node with no Anthropic SDK
    installed can still build the Ollama captioner — and so that ``core``
    imports on a node that has neither.
    """
    key = name.strip().lower()
    if key == "ollama":
        from .ollama import OllamaCaptioner

        return OllamaCaptioner(**options)
    if key == "joycaption":
        from .joycaption import JoyCaptionCaptioner

        return JoyCaptionCaptioner(**options)
    if key == "claude":
        from .claude import ClaudeCaptioner

        return ClaudeCaptioner(**options)
    raise CaptionerError(f"unknown captioner {name!r}; available: {', '.join(names())}")


def from_config(settings: Any, **overrides: Any) -> Captioner:
    """Build the configured captioner from a :class:`CaptionerConfig`.

    The one place that knows which config field feeds which backend, so
    adding a backend does not mean editing every caller.
    """
    provider = str(overrides.pop("provider", settings.provider))
    key = provider.strip().lower()
    if key == "ollama":
        options: dict[str, Any] = {
            "url": settings.ollama_url,
            "model": settings.ollama_model,
            "timeout": settings.timeout,
        }
    elif key == "joycaption":
        options = {
            "model": settings.joycaption_model,
            "quantize": settings.joycaption_quantize,
        }
    elif key == "claude":
        options = {"model": settings.claude_model, "timeout": settings.timeout}
    else:
        raise CaptionerError(f"unknown captioner {provider!r}; available: {', '.join(names())}")
    return get_captioner(key, **{**options, **overrides})


def names() -> tuple[str, ...]:
    return ("ollama", "joycaption", "claude")


def available() -> dict[str, bool]:
    """Which captioners could be *built* right now, for ``GET /node``.

    Deliberately not whether they would work: probing Ollama means a
    network round trip and probing Claude means an API call, and a node
    description should not do either. ``Captioner.test()`` answers that,
    when the operator asks it to.
    """
    return {
        "ollama": True,
        "joycaption": _importable("transformers") and _importable("torch"),
        "claude": _importable("anthropic"),
    }


def _importable(module: str) -> bool:
    """Is the package there, without paying to import it?

    ``find_spec`` reads the metadata; importing torch to find out costs
    seconds and a CUDA context, on an endpoint that describes a node.
    """
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def labels() -> dict[str, str]:
    """Backend name to display label, for the settings UI."""
    from .claude import ClaudeCaptioner
    from .joycaption import JoyCaptionCaptioner
    from .ollama import OllamaCaptioner

    return {
        OllamaCaptioner.name: OllamaCaptioner.label,
        JoyCaptionCaptioner.name: JoyCaptionCaptioner.label,
        ClaudeCaptioner.name: ClaudeCaptioner.label,
    }


__all__ = [
    "DEFAULT_BACKEND",
    "DEFAULT_PROMPT",
    "Captioner",
    "CaptionerError",
    "available",
    "from_config",
    "get_captioner",
    "labels",
    "names",
]
