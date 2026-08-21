"""Caption a dataset with a vision model.

The operation v1 had spread across its captioner backends and its GUI:
each backend knew how to describe one image, and the loop over a folder
lived in the window that started it - so captioning could not be
scripted, queued, or run over SSH. Here the loop is the operation and the
backend is an argument, like every other op in this package.

Three behaviours worth calling out, all of them lessons from watching a
200-image set fail on image three:

* **The backend is probed once, before the loop.** A stopped Ollama
  daemon should be one clear message, not two hundred identical ones.
* **Consecutive failures abort.** Beyond a handful in a row it is the
  backend that is broken, not the images, and grinding through the rest
  wastes an hour to learn nothing.
* **A refusal is a per-image result, not a crash.** Vision models decline
  ordinary reference photography often enough that it has to be an
  outcome the batch reports and carries on from.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ... import paths
from ...captioners import Captioner
from ...captioners.base import DEFAULT_PROMPT
from ...events import Emitter, Log, Progress, is_cancelled, no_op, safe
from ..item import DatasetItem
from ..scan import scan

#: Failures in a row that mean the backend is gone rather than the image
#: being awkward. Low enough to fail fast, high enough to ride out a few
#: genuine refusals in a row.
ABORT_AFTER = 5


@dataclass
class CaptionResult:
    """What happened. ``items`` describes the dataset after the operation."""

    root: Path
    captioned: int = 0
    #: Items left alone because they already had a caption.
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    #: Set when the run stopped early - a dead backend or a cancel.
    aborted: str = ""
    items: list[DatasetItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.aborted

    @property
    def total(self) -> int:
        return self.captioned + self.skipped + len(self.failed)

    def summary(self) -> str:
        parts = [f"{self.captioned} captioned"]
        if self.skipped:
            parts.append(f"{self.skipped} already had one")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if self.aborted:
            parts.append(f"stopped: {self.aborted}")
        return ", ".join(parts)

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root.as_posix(),
            "captioned": self.captioned,
            "skipped": self.skipped,
            "failed": [{"stem": stem, "reason": reason} for stem, reason in self.failed],
            "aborted": self.aborted,
            "ok": self.ok,
        }


def caption(
    root: str | os.PathLike[str],
    captioner: Captioner,
    *,
    items: Sequence[DatasetItem] | None = None,
    prompt: str = DEFAULT_PROMPT,
    prefix: str = "",
    overwrite: bool = False,
    max_tokens: int = 400,
    extensions: Iterable[str] | None = None,
    caption_ext: str = ".txt",
    abort_after: int = ABORT_AFTER,
    emit: Emitter = no_op,
    cancel: threading.Event | None = None,
) -> CaptionResult:
    """Write a ``.txt`` caption beside every image that lacks one.

    *items* restricts the run to a subset - the gallery's "caption
    selection". *overwrite* re-captions items that already have text;
    without it an existing caption always wins, because a caption someone
    edited by hand is worth more than anything a model will write.

    *prefix* is prepended to every caption, which is where a trigger token
    goes. It is applied here rather than asked of the model, because a
    model asked to begin with a specific token will sometimes decline to.
    """
    emit = safe(emit)
    source = paths.expand(root)
    result = CaptionResult(root=source)

    selected = list(items) if items is not None else scan(
        source, extensions=extensions, caption_ext=caption_ext, cancel=cancel
    )
    if is_cancelled(cancel):
        # `scan` honours the same token, so a cancel before the first image
        # arrives here as an empty list. Saying "no images found" then would
        # be a lie about the dataset.
        result.aborted = "cancelled before it started"
        emit(Log(line=result.aborted, level="warning"))
        return result
    if not selected:
        emit(Log(line=f"No images found in {source}", level="warning"))
        return result

    # Probe once. Two hundred copies of "cannot reach Ollama" is not a
    # more informative error than one.
    ready, message = captioner.test()
    if not ready:
        result.aborted = message
        emit(Log(line=message, level="error"))
        return result
    emit(Log(line=message, level="info"))

    total = len(selected)
    emit(Progress(step=0, total=total, message="Captioning"))
    consecutive = 0

    for index, item in enumerate(selected, start=1):
        if is_cancelled(cancel):
            result.aborted = f"cancelled after {index - 1} of {total}"
            emit(Log(line=result.aborted, level="warning"))
            break

        if item.has_caption() and not overwrite:
            result.skipped += 1
            result.items.append(item)
            emit(Progress(step=index, total=total, message="Captioning"))
            continue

        ok, text = captioner.describe(item.image, prompt, max_tokens)
        if not ok:
            consecutive += 1
            result.failed.append((item.stem, text))
            result.items.append(item)
            emit(Log(line=f"{item.image.name}: {text}", level="error"))
            if abort_after and consecutive >= abort_after:
                result.aborted = (
                    f"{consecutive} failures in a row - the captioner looks broken, "
                    f"stopping at {index} of {total}"
                )
                emit(Log(line=result.aborted, level="error"))
                break
            emit(Progress(step=index, total=total, message="Captioning"))
            continue

        consecutive = 0
        final = _clean(text, prefix)
        try:
            result.items.append(item.write_caption(final, caption_ext))
        except OSError as exc:
            result.failed.append((item.stem, str(exc)))
            result.items.append(item)
            emit(Log(line=f"{item.image.name}: {exc}", level="error"))
        else:
            result.captioned += 1
            emit(Log(line=f"{item.image.name}: {final}", level="debug"))

        emit(Progress(step=index, total=total, message="Captioning"))

    # Anything the loop never reached still belongs in the description of
    # the dataset it leaves behind.
    seen = {done.stem for done in result.items}
    result.items.extend(item for item in selected if item.stem not in seen)

    emit(Log(line=result.summary(), level="info" if result.ok else "warning"))
    return result


#: Openers vision models fall back on however plainly they are asked not
#: to. Stripped rather than re-prompted, because re-prompting costs a
#: second call and works about half the time.
PREAMBLES = (
    "this image shows",
    "this image depicts",
    "the image shows",
    "the image depicts",
    "in this image,",
    "in this image",
    "here is a description of the image:",
    "sure, here is a description of the image:",
    "caption:",
)


def _clean(text: str, prefix: str = "") -> str:
    """Tidy a model's answer into something usable as a caption."""
    body = " ".join(text.split()).strip().strip('"')

    lowered = body.lower()
    for preamble in PREAMBLES:
        if lowered.startswith(preamble):
            body = body[len(preamble) :].lstrip(" :,")
            body = body[:1].upper() + body[1:]
            break

    # A trigger token written as "mara," and one written as "mara" have to
    # produce the same caption, so the separator is normalised rather than
    # trusted from whichever way it happened to be typed.
    prefix = prefix.strip().rstrip(",;:").strip()
    if not prefix:
        return body
    if not body:
        return prefix
    return f"{prefix}, {body}"


__all__ = ["ABORT_AFTER", "CaptionResult", "caption"]
