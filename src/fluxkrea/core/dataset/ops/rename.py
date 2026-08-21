"""Mass rename. Was v1's ``mass_rename_images``.

Two things make this the most dangerous operation in the application: it
is destructive and in place, and it must move a whole bundle atomically or
the image-to-caption-to-mask pairing breaks silently.

Carried forward from v1 (doc 01, ``d1890ce``):

* **Two phases.** Every file moves to a unique temp name, then temp to
  final. A target that collides with another file *in the same batch* is
  then a non-issue, which is what lets a folder be renamed onto its own
  prefix.
* **A journal and a rollback.** A failure or a cancel part way through
  restores every move, rather than leaving the folder half renamed.
* **``start_index`` continues past bystanders.** Renaming a selection
  picks up after whatever same-prefix files are already in the folder,
  instead of restarting at 1 and aborting on the first collision.

New here: the plan is a first-class value. ``plan_rename`` computes the
whole mapping and its conflicts without touching the disk, so a caller -
the CLI's ``--dry-run``, the API's preview, the gallery - can show it
before anything moves. And every move is a *bundle* move: the caption and
the mask go with the image because :class:`DatasetItem` says they belong
to it, not because this function remembered to handle them.
"""

from __future__ import annotations

import os
import random
import re
import string
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ... import paths
from ...events import Emitter, Log, Progress, is_cancelled, no_op, safe
from ..item import DatasetItem
from ..metadata import Metadata
from ..scan import scan

#: Temp names used between the two phases. Distinctive enough that one left
#: behind by a hard kill is obviously ours.
TEMP_PREFIX = "__fk_rename_"

_INDEX_PATTERN = "_[a-z]?(\\d+)$"


@dataclass(frozen=True, slots=True)
class Move:
    """One bundle moving from *before* to *after*."""

    before: DatasetItem
    after: DatasetItem

    @property
    def changed(self) -> bool:
        return self.before.image.name != self.after.image.name

    def pairs(self) -> list[tuple[Path, Path]]:
        """Member-by-member source and destination, image first."""
        pairs = [(self.before.image, self.after.image)]
        if self.before.caption and self.after.caption:
            pairs.append((self.before.caption, self.after.caption))
        if self.before.mask and self.after.mask:
            pairs.append((self.before.mask, self.after.mask))
        return pairs


@dataclass
class RenamePlan:
    """The complete mapping, computed without touching the disk."""

    root: Path
    prefix: str
    start_index: int
    digits: int
    moves: list[Move] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    scrambled: bool = False

    @property
    def ok(self) -> bool:
        return not self.conflicts

    @property
    def mapping(self) -> dict[str, str]:
        """Old image filename -> new image filename. What metadata follows."""
        return {m.before.image.name: m.after.image.name for m in self.moves}

    def summary(self) -> str:
        if not self.ok:
            return f"refused: {'; '.join(self.conflicts[:3])}"
        moving = sum(1 for m in self.moves if m.changed)
        return f"{moving} of {len(self.moves)} bundles would move, from {self.prefix}_{str(self.start_index).zfill(self.digits)}"

    def describe(self, limit: int = 0) -> list[str]:
        lines = [f"{m.before.image.name} -> {m.after.image.name}" for m in self.moves]
        return lines[:limit] if limit else lines

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root.as_posix(),
            "prefix": self.prefix,
            "start_index": self.start_index,
            "digits": self.digits,
            "scrambled": self.scrambled,
            "ok": self.ok,
            "conflicts": self.conflicts,
            "moves": [
                {
                    "before": m.before.image.name,
                    "after": m.after.image.name,
                    "sidecars": len(m.pairs()) - 1,
                }
                for m in self.moves
            ],
        }


@dataclass
class RenameResult:
    plan: RenamePlan
    renamed: int = 0
    sidecars: int = 0
    rolled_back: bool = False
    error: str = ""
    items: list[DatasetItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.error

    def summary(self) -> str:
        if self.error:
            state = "rolled back, no files changed" if self.rolled_back else "failed"
            return f"Rename {state}: {self.error}"
        return f"{self.renamed} images renamed, {self.sidecars} sidecars followed"

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "renamed": self.renamed,
            "sidecars": self.sidecars,
            "rolled_back": self.rolled_back,
            "error": self.error,
            "plan": self.plan.as_dict(),
        }


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def plan_rename(
    root: str | os.PathLike[str],
    prefix: str,
    *,
    items: Sequence[DatasetItem] | None = None,
    scramble: bool = False,
    start_index: int | None = None,
    digits: int | None = None,
    extensions: Iterable[str] | None = None,
    caption_ext: str = ".txt",
    seed: int | None = None,
    cancel: threading.Event | None = None,
) -> RenamePlan:
    """Compute the whole rename without moving anything.

    ``start_index`` of ``None`` picks the lowest number that will not
    collide with same-prefix files *outside* this batch - so renaming a
    whole folder onto its own prefix still starts at 1, while renaming a
    selection continues after what is already there.
    """
    folder = paths.expand(root)
    prefix = prefix.strip()
    if not prefix:
        raise ValueError("rename needs a prefix")
    if _unsafe(prefix):
        raise ValueError(f"prefix contains characters that are not safe in a filename: {prefix!r}")

    selected = list(items) if items is not None else scan(
        folder, extensions=extensions, caption_ext=caption_ext, cancel=cancel
    )

    width = digits if digits is not None else max(3, len(str(len(selected))))
    first = start_index if start_index is not None else _next_free_index(folder, prefix, selected)
    scramble_chars = _scramble_characters(len(selected), seed) if scramble else []

    plan = RenamePlan(
        root=folder,
        prefix=prefix,
        start_index=first,
        digits=width,
        scrambled=scramble,
    )

    for offset, item in enumerate(selected):
        number = str(first + offset).zfill(width)
        stem = f"{prefix}_{scramble_chars[offset]}{number}" if scramble else f"{prefix}_{number}"
        plan.moves.append(Move(before=item, after=item.renamed_to(stem)))

    plan.conflicts = _conflicts(folder, plan, selected)
    return plan


def _conflicts(folder: Path, plan: RenamePlan, selected: Sequence[DatasetItem]) -> list[str]:
    """Targets that already exist and are *not* part of this batch.

    A target held by a file being renamed is fine - the temp phase clears it
    out first. Only bystanders are fatal.
    """
    batch: set[Path] = set()
    for item in selected:
        batch.update(p.resolve() for p in item.members())

    clashes: list[str] = []
    claimed: dict[Path, str] = {}

    for move in plan.moves:
        for before, after in move.pairs():
            resolved = after.resolve()
            if resolved == before.resolve():
                continue
            if after.exists() and resolved not in batch:
                clashes.append(f"{after.name} already exists")
            owner = claimed.get(resolved)
            if owner is not None:
                clashes.append(f"{after.name} is claimed by both {owner} and {move.before.stem}")
            claimed[resolved] = move.before.stem

    return sorted(set(clashes))


def _next_free_index(folder: Path, prefix: str, batch: Sequence[DatasetItem]) -> int:
    """Lowest number that will not collide with existing ``prefix_NNN`` files."""
    batch_stems = {item.stem.lower() for item in batch}
    pattern = re.compile(re.escape(prefix) + _INDEX_PATTERN, re.IGNORECASE)

    try:
        names = list(folder.iterdir())
    except OSError:
        return 1

    highest = 0
    for entry in names:
        stem = entry.stem
        if stem.lower() in batch_stems:
            continue
        match = pattern.match(stem)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def _scramble_characters(count: int, seed: int | None) -> list[str]:
    """A letter in front of each number, to break alphabetical order.

    Trainers that read a folder in name order otherwise see the dataset in
    whatever order it was captured in, which correlates batches with scenes.
    """
    rng = random.Random(seed)
    letters = list(string.ascii_lowercase)
    rng.shuffle(letters)

    chosen: list[str] = []
    index = 0
    for _ in range(count):
        if index >= len(letters):
            rng.shuffle(letters)
            index = 0
        chosen.append(letters[index])
        index += 1
    return chosen


def _unsafe(value: str) -> bool:
    """Reject anything that is not a filename on both Windows and Linux."""
    return bool(set(value) & set('<>:"/\\|?*')) or any(ord(c) < 32 for c in value)


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def execute(
    plan: RenamePlan,
    *,
    emit: Emitter = no_op,
    cancel: threading.Event | None = None,
    metadata: Metadata | None = None,
) -> RenameResult:
    """Apply a plan, in two phases, with a rollback on any failure."""
    emit = safe(emit)
    result = RenameResult(plan=plan)

    if not plan.ok:
        result.error = "; ".join(plan.conflicts[:5])
        emit(Log(line=f"Rename refused: {result.error}", level="error"))
        return result

    moves = [m for m in plan.moves if m.changed]
    if not moves:
        emit(Log(line="Nothing to rename - every name is already correct", level="info"))
        result.items = [m.after for m in plan.moves]
        return result

    total = len(moves) * 2
    journal: list[tuple[Path, Path]] = []

    def move(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        journal.append((source, destination))

    def rollback() -> None:
        for source, destination in reversed(journal):
            try:
                destination.rename(source)
            except OSError as exc:
                emit(Log(line=f"Rollback failed for {destination.name}: {exc}", level="error"))

    emit(Progress(step=0, total=total, message="Renaming"))
    staged: list[list[tuple[Path, Path]]] = []
    cancelled = False

    try:
        # Phase 1: everything out of the way, so no target is occupied.
        for index, item_move in enumerate(moves):
            if is_cancelled(cancel):
                cancelled = True
                break
            pairs: list[tuple[Path, Path]] = []
            for member, (source, destination) in enumerate(item_move.pairs()):
                temp = source.with_name(f"{TEMP_PREFIX}{index}_{member}__{source.suffix}")
                move(source, temp)
                pairs.append((temp, destination))
            staged.append(pairs)
            emit(Progress(step=index + 1, total=total, message="Renaming"))

        # Phase 2: temp names to final names. Deliberately not cancellable -
        # the files are sitting under temp names and must be resolved.
        if not cancelled:
            for index, pairs in enumerate(staged):
                for member, (temp, destination) in enumerate(pairs):
                    move(temp, destination)
                    if member == 0:
                        result.renamed += 1
                    else:
                        result.sidecars += 1
                emit(Progress(step=len(moves) + index + 1, total=total, message="Renaming"))

    except OSError as exc:
        rollback()
        result.error = str(exc)
        result.rolled_back = True
        emit(Log(line=result.summary(), level="error"))
        return result

    if cancelled:
        rollback()
        result.error = "cancelled"
        result.rolled_back = True
        emit(Log(line="Rename cancelled. No files were changed.", level="warning"))
        return result

    meta = metadata if metadata is not None else Metadata.load(plan.root)
    if len(meta):
        meta.apply_rename(plan.mapping)
        meta.save()

    result.items = [m.after for m in plan.moves]
    emit(Log(line=result.summary(), level="info"))
    return result


def rename(
    root: str | os.PathLike[str],
    prefix: str,
    *,
    items: Sequence[DatasetItem] | None = None,
    scramble: bool = False,
    start_index: int | None = None,
    digits: int | None = None,
    extensions: Iterable[str] | None = None,
    caption_ext: str = ".txt",
    seed: int | None = None,
    emit: Emitter = no_op,
    cancel: threading.Event | None = None,
) -> RenameResult:
    """Plan and execute in one call, for callers that do not want a preview."""
    plan = plan_rename(
        root,
        prefix,
        items=items,
        scramble=scramble,
        start_index=start_index,
        digits=digits,
        extensions=extensions,
        caption_ext=caption_ext,
        seed=seed,
        cancel=cancel,
    )
    return execute(plan, emit=emit, cancel=cancel)
