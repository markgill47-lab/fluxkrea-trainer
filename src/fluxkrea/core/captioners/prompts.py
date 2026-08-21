"""Saved caption prompts.

A prompt is the difference between "a woman in a room" and a caption a
LoRA can actually learn from, and the good one is arrived at by editing
it a dozen times against a real dataset. v1 had one prompt box and no
memory, so every improvement was lost the moment you changed it — which
in practice means everyone retypes an approximation of last week's
prompt.

Kept as its own file rather than in ``config.toml``. The config is a
fixed set of typed settings; this is a growing collection of paragraphs,
and mixing them makes the config file unreadable and this list awkward to
edit. Same directory, so the two travel together.

**Built-ins cannot be deleted, only shadowed.** Saving over a built-in
name keeps your version and remembers the original underneath; deleting
yours brings the original back. The alternative — letting a built-in be
destroyed — means a bad edit costs you a prompt you cannot get back
without reinstalling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import paths
from .base import DEFAULT_PROMPT

PROMPTS_FILENAME = "prompts.json"

#: Every prompt that lists what to cover needs this, and finding out why
#: cost two captions out of a batch of forty-two: asked to cover "pose,
#: expression, framing, clothing", JoyCaption sometimes reads the list as a
#: form and returns "**Pose:** Standing. **Expression:** Neutral." A LoRA
#: trained on that learns the labels.
PROSE = (
    "Write one flowing paragraph of prose. Do not use headings, labels, "
    "bullet points, bold text or any other formatting."
)

#: Shipped prompts. Written for LoRA training captions, where the caption
#: teaches the model what varies between images — so anything true of
#: *every* image in the set is better left out of it.
BUILTIN: dict[str, str] = {
    "default": DEFAULT_PROMPT,
    "person": (
        "Describe this photograph of a person for a training caption. Cover "
        "the pose, expression, camera angle, framing, clothing, hair, setting "
        "and lighting. Do not describe facial features or identity - those are "
        "what the model is being taught. Do not editorialise and do not "
        "mention image quality. " + PROSE
    ),
    "clothing": (
        "Describe the garment in this image for a training caption. Cover its "
        "type, cut, fabric, colour, pattern, fastenings and how it sits on the "
        "body, then the pose and framing. Keep the description of the wearer "
        "brief. " + PROSE
    ),
    "style": (
        "Describe this image for a style training caption. State the subject "
        "plainly in a few words, then concentrate on medium, technique, "
        "palette, line quality, texture, lighting and composition. The subject "
        "matters less than how it is rendered. " + PROSE
    ),
    "terse": (
        "Describe this image in one sentence for a training caption. Subject, "
        "pose, setting, lighting. No preamble, no editorialising."
    ),
}


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str
    text: str
    #: True when this is a shipped prompt with no saved version over it.
    builtin: bool
    #: True when a saved prompt is standing over a shipped one of the same
    #: name. The UI offers "revert" rather than "delete" for these.
    shadows_builtin: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "text": self.text,
            "builtin": self.builtin,
            "shadows_builtin": self.shadows_builtin,
        }


class PromptLibrary:
    """The saved prompts on one node, backed by a JSON file."""

    def __init__(self, file: Path | None = None) -> None:
        self.file = file or (paths.config_dir() / PROMPTS_FILENAME)
        self._saved: dict[str, str] = {}
        self._loaded = False

    # -- storage -----------------------------------------------------------

    def load(self) -> PromptLibrary:
        """Read the file. A corrupt one is ignored, not fatal.

        Losing a prompt list is annoying; refusing to caption because of it
        would be worse. Contrast ``face_boxes.json``, which holds human
        review work and does raise.
        """
        self._loaded = True
        if not self.file.is_file():
            return self
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self
        if isinstance(data, dict):
            prompts = data.get("prompts", data)
            if isinstance(prompts, dict):
                self._saved = {
                    str(name): str(text)
                    for name, text in prompts.items()
                    if str(name).strip() and str(text).strip()
                }
        return self

    def _ensure(self) -> None:
        if not self._loaded:
            self.load()

    def save_file(self) -> Path:
        paths.ensure_dir(self.file.parent)
        payload = json.dumps({"prompts": self._saved}, indent=2, ensure_ascii=False)
        # Written through a temp file: a half-written prompt list read back
        # as corrupt would silently empty the library.
        tmp = self.file.with_suffix(self.file.suffix + ".tmp")
        tmp.write_text(payload + "\n", encoding="utf-8")
        tmp.replace(self.file)
        return self.file

    # -- reading -----------------------------------------------------------

    def all(self) -> list[Prompt]:
        """Every prompt, built-ins first, each saying where it came from."""
        self._ensure()
        out = [
            Prompt(
                name=name,
                text=self._saved.get(name, text),
                builtin=name not in self._saved,
                shadows_builtin=name in self._saved,
            )
            for name, text in BUILTIN.items()
        ]
        out.extend(
            Prompt(name=name, text=text, builtin=False)
            for name, text in sorted(self._saved.items())
            if name not in BUILTIN
        )
        return out

    def get(self, name: str) -> str | None:
        """The text for a name, saved version winning over the built-in."""
        self._ensure()
        key = name.strip()
        if key in self._saved:
            return self._saved[key]
        return BUILTIN.get(key)

    def resolve(self, name: str | None, fallback: str = "") -> str:
        """Prompt text for a name, falling back to *fallback* then the default.

        The one place that decides what an unknown name means: the default
        prompt, not an empty one. A typo in a prompt name should produce
        ordinary captions, not two hundred images described by nothing.
        """
        if name and name.strip():
            found = self.get(name)
            if found:
                return found
        return fallback.strip() or DEFAULT_PROMPT

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    # -- writing -----------------------------------------------------------

    def save(self, name: str, text: str) -> Prompt:
        """Add or replace a saved prompt, and write the file."""
        self._ensure()
        key = name.strip()
        body = text.strip()
        if not key:
            raise ValueError("a prompt needs a name")
        if not body:
            raise ValueError("a prompt needs some text")
        self._saved[key] = body
        self.save_file()
        return Prompt(
            name=key, text=body, builtin=False, shadows_builtin=key in BUILTIN
        )

    def delete(self, name: str) -> bool:
        """Remove a saved prompt. Returns whether anything was removed.

        A built-in that was shadowed reappears; a built-in that was never
        shadowed cannot be deleted at all.
        """
        self._ensure()
        key = name.strip()
        if key not in self._saved:
            return False
        del self._saved[key]
        self.save_file()
        return True

    def names(self) -> list[str]:
        return [prompt.name for prompt in self.all()]


#: Exported under a fuller name; ``BUILTIN`` reads better in here.
BUILTIN_PROMPTS = BUILTIN

__all__ = [
    "BUILTIN",
    "BUILTIN_PROMPTS",
    "PROMPTS_FILENAME",
    "Prompt",
    "PromptLibrary",
]
