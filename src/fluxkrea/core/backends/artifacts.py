"""Finding the LoRA a run produced, and putting it where ComfyUI looks.

A finished run leaves a folder of checkpoints, samples and a generated
config, and the thing anybody actually wants out of it is one
``.safetensors`` file. Two questions follow, and both were being answered
by hand:

**Which file is the LoRA?** ai-toolkit rotates checkpoints - three kept
plus the final one - so the folder holds four files with the same stem and
different step numbers. The final one is the answer, and "final" is the
one *without* a step number in its name, because that is how ai-toolkit
distinguishes it. Ordering by mtime instead would be right most of the
time and wrong exactly when a sample render touched a file last.

**Where does it go?** Under ComfyUI's ``models/loras/<family>``, where the
family comes off the model record (``Model.lora_dir``) rather than out of
a substring test on the filename. v1 inferred architectures from names and
that is how a checkpoint reached the wrong trainer; inferring a
*destination* the same way would put a Krea 2 LoRA in the FLUX.2 folder,
where it loads, produces noise, and looks like a training failure.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import paths
from .models import Model, find as find_model

#: What a trained LoRA is written as. Nothing else in a run folder is one.
LORA_SUFFIX = ".safetensors"

#: ai-toolkit names rotated checkpoints ``<name>_000000500.safetensors`` and
#: the final one ``<name>.safetensors``. The digits are what tell them apart.
STEP_PATTERN = re.compile(r"_(\d{4,})$")

#: Where ComfyUI keeps LoRAs, relative to its install root.
LORA_ROOT = ("models", "loras")


class PublishError(Exception):
    """A publish that will not happen, with the reason a person can act on."""


@dataclass(frozen=True, slots=True)
class Artifact:
    """One ``.safetensors`` file a run produced."""

    path: Path
    #: Step it was saved at, or ``None`` for the final weights.
    step: int | None
    size: int
    mtime: float

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def final(self) -> bool:
        return self.step is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path.as_posix(),
            "step": self.step,
            "final": self.final,
            "size": self.size,
            "mtime": self.mtime,
        }


def find_artifacts(output: str | os.PathLike[str] | None) -> list[Artifact]:
    """Every LoRA in a run's output folder, final first then newest step.

    Searched one level deep as well as at the top: ai-toolkit appends the
    job name to ``training_folder`` itself, and a run configured before
    that was understood still has its weights a directory down.
    """
    if not output:
        return []
    root = paths.expand(output)
    if not root.is_dir():
        return []

    found: list[Artifact] = []
    seen: set[Path] = set()
    folders = [root, *(child for child in _safe_iterdir(root) if child.is_dir())]
    for folder in folders:
        for entry in _safe_iterdir(folder):
            if not entry.is_file() or entry.suffix.lower() != LORA_SUFFIX:
                continue
            try:
                resolved = entry.resolve()
            except OSError:
                resolved = entry
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                stat = entry.stat()
            except OSError:
                continue
            found.append(
                Artifact(
                    path=entry,
                    step=_step_of(entry.stem),
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                )
            )

    # Final first, then the highest step - the order somebody wants them
    # offered in, which is not the order the filesystem hands them over.
    found.sort(key=lambda a: (0 if a.final else 1, -(a.step or 0), a.name))
    return found


def final_artifact(output: str | os.PathLike[str] | None) -> Artifact | None:
    """The LoRA a run is *for*, or the newest checkpoint if it never finished."""
    found = find_artifacts(output)
    return found[0] if found else None


def lora_family(model_id: str) -> str:
    """Which ``models/loras/<here>`` folder this model's LoRAs belong in.

    Read off the model record, never inferred from the run or file name.
    An unknown model raises: putting a LoRA somewhere plausible is worse
    than refusing, because a file in the wrong family folder loads and
    produces noise rather than an error.
    """
    model: Model | None = find_model(model_id)
    if model is None:
        raise PublishError(
            f"unknown model {model_id!r}, so there is no folder to publish its "
            "LoRA into. Publishing to a guessed folder would produce a LoRA that "
            "loads and generates noise, which reads as a bad training run."
        )
    if not model.lora_dir:
        raise PublishError(f"{model.id} has no publish folder configured")
    return model.lora_dir


def lora_destination(comfyui: str | os.PathLike[str] | None, model_id: str) -> Path:
    """The folder a LoRA for *model_id* is published into."""
    if not comfyui:
        raise PublishError(
            "backends.comfyui_path is not set, so there is nowhere to publish to. "
            "Set it in config.toml on this node and restart the daemon."
        )
    root = paths.expand(comfyui)
    if not root.is_dir():
        raise PublishError(f"backends.comfyui_path points at {root.as_posix()}, which is not there")
    return root.joinpath(*LORA_ROOT, lora_family(model_id))


def publish(
    source: str | os.PathLike[str],
    comfyui: str | os.PathLike[str] | None,
    model_id: str,
    *,
    name: str = "",
    overwrite: bool = False,
) -> Path:
    """Copy a trained LoRA into ComfyUI's loras folder. Returns the target.

    A copy rather than a move or a symlink. The run folder is the record of
    what happened and stays intact; a symlink would break the moment
    somebody tidied the output directory, and on Windows it needs a
    privilege a student does not have.

    Refuses to overwrite by default. Two students publishing runs that
    happen to derive the same name is the expected collision here, and
    silently replacing one person's LoRA with another's is the worst
    available outcome.
    """
    origin = paths.expand(source)
    if not origin.is_file():
        raise PublishError(f"{origin.as_posix()} is not there to publish")
    if origin.suffix.lower() != LORA_SUFFIX:
        raise PublishError(f"{origin.name} is not a {LORA_SUFFIX} file")

    folder = lora_destination(comfyui, model_id)
    filename = _clean_name(name) if name else origin.name
    target = folder / filename

    if target.exists() and not overwrite:
        raise PublishError(
            f"{target.as_posix()} already exists. Rename the run, or publish with "
            "overwrite if replacing it is what you meant."
        )

    try:
        paths.ensure_dir(folder)
        # To a temp name in the destination folder first, then renamed:
        # ComfyUI scans that folder, and a 92MB copy in progress is a file
        # it can find, list and fail to load halfway through.
        staging = target.with_name(f".{target.name}.{os.getpid()}.partial")
        shutil.copy2(origin, staging)
        staging.replace(target)
    except OSError as exc:
        raise PublishError(f"cannot publish to {target.as_posix()}: {exc}") from exc
    return target


def _clean_name(name: str) -> str:
    """A caller-supplied filename, with any path in it removed.

    The name arrives over HTTP. ``../../autoexec`` must land as a file
    called ``autoexec`` in the loras folder, not anywhere else.
    """
    stem = Path(str(name).replace("\\", "/")).name.strip()
    if not stem:
        raise PublishError("that is not a usable filename")
    return stem if stem.lower().endswith(LORA_SUFFIX) else f"{stem}{LORA_SUFFIX}"


def _step_of(stem: str) -> int | None:
    match = STEP_PATTERN.search(stem)
    return int(match.group(1)) if match else None


def _safe_iterdir(folder: Path) -> list[Path]:
    try:
        return sorted(folder.iterdir(), key=lambda p: p.name)
    except OSError:
        return []


__all__ = [
    "LORA_ROOT",
    "LORA_SUFFIX",
    "Artifact",
    "PublishError",
    "final_artifact",
    "find_artifacts",
    "lora_destination",
    "lora_family",
    "publish",
]
