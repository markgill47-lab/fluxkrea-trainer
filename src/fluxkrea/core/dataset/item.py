"""``DatasetItem`` - the invariant that started the rewrite.

A training example is not a file. It is a bundle::

    punch_014.jpg          the image
    punch_014.txt          the caption
    masks/punch_014.png    the loss mask   (optional)

Every operation that touches one member must touch all of them, or the
bundle silently desynchronises. v1 has no such concept: ``fix_images``,
``create_duplicates`` and ``mass_rename_images`` each hand-roll their own
sidecar handling. Three copies, three chances to forget - and the failure
is silent, because ai-toolkit sets ``has_mask_image`` only if it finds a
basename match and otherwise trains the image unmasked without warning
(doc 03).

So: **this module is the only place that knows what a bundle contains.**
Add a fourth member later - control images, depth maps - by extending
:data:`SIDECAR_FIELDS` and the resolvers below, and every operation gets
it for free.

The type is frozen. Operations are pure mappings from old items to new
ones; execution against the filesystem is a separate step, which is what
makes a rename plannable, previewable and rollbackable.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from .. import paths

#: The sidecar attributes of a bundle, in the order operations should
#: handle them. Extending this tuple - and :meth:`DatasetItem.rebased` -
#: is the entire cost of adding a new bundle member.
SIDECAR_FIELDS = ("caption", "mask")

#: Captions are ``.txt`` sidecars, because that is what every trainer in
#: use actually reads (``caption_ext: txt``). Doc 03, "caption storage".
CAPTION_SUFFIX = ".txt"

#: Masks are 8-bit greyscale PNG, never JPEG: JPEG ringing on a hard
#: black/white edge produces grey fringes, and a grey pixel in a mask is a
#: partial loss weight (doc 03).
MASK_SUFFIX = ".png"

PREVIEW_SUFFIX = ".jpg"


@dataclass(frozen=True, slots=True)
class DatasetItem:
    """One training example: an image and the sidecars that belong to it."""

    image: Path
    caption: Path | None = None
    mask: Path | None = None
    #: good / ok / bad, from the captioner. Cached in ``metadata.json``,
    #: never authoritative for anything the trainer reads.
    quality: str | None = None

    # -- identity ---------------------------------------------------------

    @property
    def stem(self) -> str:
        """The basename that ties the bundle together."""
        return self.image.stem

    @property
    def root(self) -> Path:
        """The dataset folder this item lives in."""
        return self.image.parent

    @property
    def suffix(self) -> str:
        return self.image.suffix

    def __str__(self) -> str:
        return self.image.name

    # -- membership -------------------------------------------------------

    def sidecars(self) -> Iterator[Path]:
        """The sidecar paths this item actually has.

        The single place that knows a bundle has a caption and a mask.
        """
        for name in SIDECAR_FIELDS:
            value = getattr(self, name)
            if value is not None:
                yield value

    def members(self) -> Iterator[Path]:
        """Every file in the bundle, image first."""
        yield self.image
        yield from self.sidecars()

    def exists(self) -> bool:
        return self.image.is_file()

    def missing(self) -> list[Path]:
        """Bundle members recorded on the item but absent from disk."""
        return [p for p in self.members() if not p.is_file()]

    # -- where sidecars *would* go ----------------------------------------

    def expected_caption(self, caption_ext: str = CAPTION_SUFFIX) -> Path:
        """Where this item's caption belongs, whether or not it exists."""
        ext = caption_ext if caption_ext.startswith(".") else f".{caption_ext}"
        return self.image.with_suffix(ext)

    def expected_mask(self) -> Path:
        """Where this item's mask belongs: ``masks/<stem>.png`` beside the image.

        Matches ai-toolkit's basename lookup in the ``mask_path`` folder
        (doc 04, "contract").
        """
        return paths.masks_dir(self.root) / f"{self.stem}{MASK_SUFFIX}"

    def expected_preview(self) -> Path:
        """Where the redacted review preview belongs. Never read by a trainer."""
        return paths.preview_dir(self.root) / f"{self.stem}{PREVIEW_SUFFIX}"

    # -- pure transformations ---------------------------------------------

    def renamed_to(self, new_stem: str) -> DatasetItem:
        """This bundle under a new basename. Pure; touches no files.

        Each member keeps its own directory and extension - the mask stays
        in ``masks/`` and stays a PNG - so the basename match the trainer
        relies on survives.
        """
        return replace(
            self,
            image=_restem(self.image, new_stem),
            caption=_restem(self.caption, new_stem),
            mask=_restem(self.mask, new_stem),
        )

    def suffixed(self, suffix: str) -> DatasetItem:
        """``punch_014`` -> ``punch_014_flipHor``. Used by augmentation."""
        return self.renamed_to(f"{self.stem}{suffix}")

    def rebased(self, new_root: str | os.PathLike[str]) -> DatasetItem:
        """The same bundle in a different dataset folder.

        The mask is rebased into the *new* folder's ``masks/``, not copied
        flat next to the image - the layout is a property of the folder,
        not of the item.
        """
        root = paths.expand(new_root)
        return replace(
            self,
            image=root / self.image.name,
            caption=None if self.caption is None else root / self.caption.name,
            mask=None if self.mask is None else paths.masks_dir(root) / self.mask.name,
        )

    def with_caption_path(self, path: Path | None) -> DatasetItem:
        return replace(self, caption=path)

    def with_mask_path(self, path: Path | None) -> DatasetItem:
        return replace(self, mask=path)

    def with_quality(self, quality: str | None) -> DatasetItem:
        return replace(self, quality=quality)

    # -- caption text ------------------------------------------------------

    def read_caption(self) -> str:
        """The caption text, or empty string. ``.txt`` is the truth (doc 03)."""
        if self.caption is None or not self.caption.is_file():
            return ""
        try:
            return self.caption.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            # v1 datasets predate any encoding discipline; salvage rather
            # than fail a whole scan over one file written by Notepad.
            return self.caption.read_text(encoding="utf-8", errors="replace").strip()

    def write_caption(self, text: str, caption_ext: str = CAPTION_SUFFIX) -> DatasetItem:
        """Write the caption sidecar, creating it if the item had none.

        Returns the updated item - the caller must keep it, or the bundle
        it holds no longer describes what is on disk.
        """
        target = self.caption or self.expected_caption(caption_ext)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.strip() + "\n", encoding="utf-8", newline="\n")
        return self.with_caption_path(target)

    def has_caption(self) -> bool:
        return self.caption is not None and self.caption.is_file()

    def has_mask(self) -> bool:
        return self.mask is not None and self.mask.is_file()


def _restem(path: Path | None, new_stem: str) -> Path | None:
    """Replace a path's basename, keeping its folder and its extension.

    ``Path.with_stem`` would do, but a filename like ``punch.014.jpg``
    makes ``with_suffix`` ambiguous, and the dataset folders in question
    do contain dotted names.
    """
    if path is None:
        return None
    return path.with_name(f"{new_stem}{path.suffix}")
