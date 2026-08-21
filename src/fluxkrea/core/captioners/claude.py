"""Claude — the remote captioner, for when quality matters more than privacy.

The trade against [Ollama][ollama] is explicit and the operator makes it:
Claude writes better captions and needs no VRAM beside a training job, but
every image goes to somebody else's servers. Ollama is the default for
that reason; this is here for the sets where it does not matter.

Unlike the rest of ``core/``, this speaks to a service through a vendor
SDK rather than through ``base.post_json``. The Anthropic SDK is the
supported way to call the Messages API — it carries retries, timeouts and
typed errors that would otherwise be reimplemented badly here — and it is
imported *inside* the constructor so that ``core`` still imports on a node
that has never heard of it. The guard test walks every core module; an
import at module scope would make ``anthropic`` a hard dependency of the
whole package for a feature most nodes never use.

.. [ollama] :mod:`fluxkrea.core.captioners.ollama`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import DEFAULT_PROMPT, Captioner, CaptionerError, encode_image, env_secret

#: The default model. Captioning a reference set is a perception task run
#: a few hundred times, so it runs at low effort rather than on a smaller
#: model — the quality difference on awkward framing is the reason to be
#: paying for this backend at all.
DEFAULT_MODEL = "claude-opus-5"

DEFAULT_EFFORT = "low"

#: A caption is a paragraph. This is a ceiling that stops a runaway
#: response, not a target.
DEFAULT_MAX_TOKENS = 400

#: Environment names checked in order. There is no config-file option on
#: purpose: doc 05 makes a secret in ``config.toml`` a hard error, and a
#: captioner is not the place to make an exception.
KEY_NAMES = ("FLUXKREA_CLAUDE_API_KEY", "ANTHROPIC_API_KEY", "CLAUDE_API_KEY")

#: The Messages API takes the media type explicitly rather than sniffing.
MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class ClaudeCaptioner(Captioner):
    name = "claude"
    label = "Claude (API)"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        effort: str = DEFAULT_EFFORT,
        timeout: float = 120.0,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on the env
            raise CaptionerError(
                "the Claude captioner needs the Anthropic SDK: pip install anthropic"
            ) from exc

        self._anthropic = anthropic
        self.model = model
        self.effort = effort

        # An unset ANTHROPIC_API_KEY does not mean there are no credentials:
        # the SDK also resolves an `ant auth login` profile. So only pass a
        # key when we actually found one, and otherwise let the SDK look.
        key = api_key or env_secret(*KEY_NAMES)
        kwargs: dict[str, Any] = {"timeout": timeout}
        if key:
            kwargs["api_key"] = key
        try:
            self._client = anthropic.Anthropic(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced as a build failure
            raise CaptionerError(f"cannot construct the Anthropic client: {exc}") from exc

    # -- probing -----------------------------------------------------------

    def test(self) -> tuple[bool, str]:
        """Check credentials and the model id without spending tokens.

        A retrieve on the Models API answers both questions the operator
        has — is the key good, is that model real — for the price of one
        request and no inference.
        """
        try:
            model = self._client.models.retrieve(self.model)
        except self._anthropic.AuthenticationError:
            return False, (
                "the Anthropic API rejected the credentials. Set one of "
                + ", ".join(KEY_NAMES)
                + " in the environment, or run `ant auth login`."
            )
        except self._anthropic.NotFoundError:
            return False, f"no such model: {self.model!r}"
        except self._anthropic.APIConnectionError as exc:
            return False, f"cannot reach the Anthropic API: {exc}"
        except self._anthropic.APIStatusError as exc:
            return False, f"Anthropic API error {exc.status_code}: {exc.message}"
        return True, f"Claude ready, model '{model.id}'."

    # -- captioning --------------------------------------------------------

    def describe(
        self,
        image: Path,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> tuple[bool, str]:
        if not image.is_file():
            return False, f"image not found: {image}"

        media_type = MEDIA_TYPES.get(image.suffix.lower())
        if media_type is None:
            return False, f"unsupported image type for the API: {image.suffix}"

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=int(max_tokens),
                output_config={"effort": self.effort},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": encode_image(image),
                                },
                            },
                            {"type": "text", "text": prompt or DEFAULT_PROMPT},
                        ],
                    }
                ],
            )
        except self._anthropic.RateLimitError as exc:
            # Returned rather than raised so the batch can pause and retry
            # this one image instead of losing the other two hundred.
            retry = exc.response.headers.get("retry-after", "60")
            return False, f"rate limited; retry after {retry}s"
        except self._anthropic.APIConnectionError as exc:
            return False, f"cannot reach the Anthropic API: {exc}"
        except self._anthropic.APIStatusError as exc:
            return False, f"Anthropic API error {exc.status_code}: {exc.message}"

        # Safety classifiers decline sometimes, and reference photography
        # is exactly the kind of material that trips them. That is a fact
        # about this image, not a broken run — say which, and carry on.
        if response.stop_reason == "refusal":
            detail = getattr(response.stop_details, "category", None) or "unspecified"
            return False, f"Claude declined to describe this image ({detail})"

        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not text:
            return False, "Claude returned an empty response"
        return True, text
