"""The captioner interface.

Ported from v1's ``captioners/``, which doc 01 singles out as the one part
of the old codebase already shaped correctly: "a clean ``Captioner`` base
with Claude, Ollama and JoyCaption backends behind a factory. Exactly the
shape everything else should have had."

Two changes on the way across:

**No ``requests``.** ``core/`` may not import an HTTP client — the guard
test enforces it, because core has to stay something a daemon wraps rather
than something that reaches out. A JSON POST to a local daemon does not
justify weakening that rule, so the HTTP here is ``urllib`` from the
standard library. It is a few more lines and one less dependency.

**Failures are returned, never raised.** A captioner that raises on a
network hiccup takes a 200-image batch down with it. Every backend returns
``(ok, text_or_reason)`` and the batch decides what to do.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

#: What a captioner is asked when nothing more specific is given. Written
#: for LoRA training data: describe the content, not the artistry.
DEFAULT_PROMPT = (
    "Describe this image for a training caption. State the subject, pose, "
    "framing, clothing, setting and lighting plainly. Do not editorialise, "
    "do not mention image quality, and do not begin with 'This image shows'."
)


class CaptionerError(Exception):
    """A captioner that cannot be built at all - bad name, missing key."""


class Captioner(ABC):
    """Anything that turns an image into a caption."""

    #: Shown in the UI and used as the config value.
    name: str = "captioner"
    label: str = "Captioner"

    @abstractmethod
    def test(self) -> tuple[bool, str]:
        """Probe whatever this backend depends on. Never raises."""

    @abstractmethod
    def describe(self, image: Path, prompt: str, max_tokens: int = 400) -> tuple[bool, str]:
        """Caption one image. Returns ``(ok, caption_or_reason)``."""

    def close(self) -> None:
        """Release resources. JoyCaption unloads VRAM here; most do nothing."""
        return None


# --------------------------------------------------------------------------
# stdlib HTTP
# --------------------------------------------------------------------------


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[bool, Any]:
    """POST JSON and decode JSON back. Returns ``(ok, body_or_message)``.

    Deliberately small: this exists so ``core`` can talk to a local model
    daemon without importing an HTTP client library (see the module
    docstring).
    """
    request = urllib.request.Request(  # noqa: S310 - the URL is operator-configured
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return True, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("error", body)
        except json.JSONDecodeError:
            detail = body[:300]
        return False, f"HTTP {exc.code}: {detail}"
    except urllib.error.URLError as exc:
        return False, f"cannot reach {url}: {exc.reason}"
    except TimeoutError:
        return False, f"timed out after {timeout:g}s"
    except json.JSONDecodeError:
        return False, "the response was not JSON"


def get_json(url: str, *, timeout: float = 5.0) -> tuple[bool, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return True, json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return False, f"cannot reach {url}: {exc.reason}"
    except TimeoutError:
        return False, f"timed out after {timeout:g}s"
    except json.JSONDecodeError:
        return False, "the response was not JSON"


def encode_image(image: Path) -> str:
    return base64.b64encode(image.read_bytes()).decode("ascii")


def env_secret(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None
