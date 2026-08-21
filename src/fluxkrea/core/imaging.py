"""Image loading, saving and resampling. The rules live here, once.

Two categories of rule, both learned the hard way:

**v1's fixes, carried forward rather than re-derived** (commit ``d1890ce``,
doc 01). EXIF orientation is baked into the pixels before any resize, or a
photo silently rotates 90 degrees on save. Files are opened inside a
context manager and loaded eagerly, or the handle stays open and blocks a
later rename on Windows. JPEG is written at quality 95 with
``subsampling=0``, because the default 4:2:0 smears exactly the fine
colour detail training data should not be throwing away.

**The mask resampling rule** (doc 03). A mask is not a photograph. Smooth
interpolation on a hard edge produces grey fringes, and a grey pixel in a
loss mask is a *partial* weight - a soft leak of the region being
excluded. Masks resample with NEAREST, and feathering is applied
deliberately at generation time, never acquired accidentally through a
resize.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

#: EXIF tag 0x0112. Named because the number appears nowhere else useful.
EXIF_ORIENTATION = 0x0112

#: Extensions that must be written as JPEG.
JPEG_SUFFIXES = frozenset({".jpg", ".jpeg", ".jfif"})

#: Full chroma resolution, matching v1's deliberate choice.
JPEG_QUALITY = 95

#: Below this on either edge, an image cannot be sensibly used or upscaled.
ABSOLUTE_MIN_EDGE = 32


class ImageError(Exception):
    """A file that cannot be used as training data. Never guessed around."""


@dataclass(frozen=True, slots=True)
class Size:
    width: int
    height: int

    @property
    def longest(self) -> int:
        return max(self.width, self.height)

    @property
    def shortest(self) -> int:
        return min(self.width, self.height)

    def as_tuple(self) -> tuple[int, int]:
        return (self.width, self.height)

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def load_oriented(path: str | os.PathLike[str]) -> tuple[Image.Image, bool]:
    """Open an image with its EXIF orientation baked into the pixels.

    Returns ``(image, was_rotated)``. Resizing produces a new image with no
    EXIF block, so an orientation tag that is never applied is silently
    lost on save and the picture ends up rotated relative to how it
    displayed everywhere else.

    Reading inside a context manager and calling ``load()`` also releases
    the OS file handle before returning, which matters on Windows where an
    open handle blocks a later rename or delete of the same file.
    """
    target = Path(path)
    try:
        with Image.open(target) as source:
            source.load()
            orientation = source.getexif().get(EXIF_ORIENTATION, 1)
            oriented = ImageOps.exif_transpose(source)
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageError(f"cannot read {target.name}: {exc}") from exc

    if oriented is None:  # exif_transpose returns None only for a None input
        raise ImageError(f"cannot read {target.name}")
    return oriented, orientation not in (1, None)


def read_size(path: str | os.PathLike[str]) -> Size:
    """Dimensions without decoding pixels, with EXIF orientation applied.

    A 90-degree orientation tag swaps width and height as far as every
    consumer is concerned, so reporting the raw header values would make a
    mask-size mismatch look real when it is not.
    """
    target = Path(path)
    try:
        with Image.open(target) as source:
            width, height = source.size
            orientation = source.getexif().get(EXIF_ORIENTATION, 1)
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageError(f"cannot read {target.name}: {exc}") from exc

    if orientation in (5, 6, 7, 8):
        width, height = height, width
    return Size(width, height)


def validate_image(path: str | os.PathLike[str], *, min_edge: int | None = None) -> tuple[bool, str]:
    """Check a file is a usable image. Returns ``(ok, message)``, never raises.

    ``min_edge`` of ``None`` skips the resolution floor - which is what an
    upscaling pass wants, since it is about to fix exactly that.
    """
    target = Path(path)
    try:
        with Image.open(target) as source:
            source.verify()  # detects truncation; consumes the file object
        size = read_size(target)
    except ImageError as exc:
        return False, str(exc)
    except (OSError, UnidentifiedImageError, SyntaxError) as exc:
        return False, f"invalid or corrupt image: {exc}"

    if size.shortest < ABSOLUTE_MIN_EDGE:
        return False, f"too small to process: {size} (minimum {ABSOLUTE_MIN_EDGE}px)"
    if min_edge is not None and size.shortest < min_edge:
        return False, f"below the resolution floor: {size} (minimum {min_edge}px)"
    return True, "ok"


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def save_image(image: Image.Image, path: str | os.PathLike[str]) -> Path:
    """Write using settings appropriate to the extension.

    JPEG gets quality 95 and no chroma subsampling. Anything else is saved
    as-is, so a PNG stays lossless.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() in JPEG_SUFFIXES:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(target, quality=JPEG_QUALITY, subsampling=0)
    else:
        image.save(target)
    return target


def save_mask(image: Image.Image, path: str | os.PathLike[str]) -> Path:
    """Write a mask as 8-bit greyscale PNG. Never JPEG (doc 03).

    JPEG ringing on a hard black/white edge produces the same grey fringes
    that the NEAREST rule exists to avoid.
    """
    target = Path(path)
    if target.suffix.lower() != ".png":
        raise ImageError(f"masks are PNG, not {target.suffix!r}: {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != "L":
        image = image.convert("L")
    image.save(target, optimize=True)
    return target


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------


def target_size_for(size: Size, target_longest: int) -> Size:
    """Dimensions after fitting the longest edge to *target*, aspect preserved."""
    if size.width >= size.height:
        width = target_longest
        height = max(1, round(size.height / size.width * target_longest))
    else:
        height = target_longest
        width = max(1, round(size.width / size.height * target_longest))
    return Size(width, height)


def resize_longest(image: Image.Image, target_longest: int) -> Image.Image:
    """Fit the longest edge to *target*, preserving aspect ratio. LANCZOS."""
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    size = Size(*image.size)
    return image.resize(target_size_for(size, target_longest).as_tuple(), Image.Resampling.LANCZOS)


def upscale_to_min(image: Image.Image, min_edge: int) -> Image.Image:
    """Grow an image until its shortest edge reaches *min_edge*. Never shrinks."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    size = Size(*image.size)
    if size.shortest >= min_edge:
        return image
    scale = min_edge / size.shortest
    grown = (max(1, round(size.width * scale)), max(1, round(size.height * scale)))
    return image.resize(grown, Image.Resampling.LANCZOS)


def resize_mask(mask: Image.Image, size: Size | tuple[int, int]) -> Image.Image:
    """Resample a mask to *size* with NEAREST. The rule from doc 03.

    NEAREST cannot invent a value that was not already in the mask, so a
    hard-edged mask stays hard and a deliberately feathered boundary keeps
    the gradient it was given. LANCZOS here would manufacture grey along
    every edge, and grey is a partial loss weight.
    """
    want = size.as_tuple() if isinstance(size, Size) else tuple(size)
    if mask.mode != "L":
        mask = mask.convert("L")
    return mask.resize(want, Image.Resampling.NEAREST)


def threshold_mask(mask: Image.Image, cutoff: int = 128) -> Image.Image:
    """Force a mask back to hard black and white."""
    if mask.mode != "L":
        mask = mask.convert("L")
    return mask.point(lambda value: 255 if value >= cutoff else 0, mode="L")


def has_grey(mask: Image.Image) -> bool:
    """True if a mask contains partial weights - a soft leak, unless deliberate."""
    if mask.mode != "L":
        mask = mask.convert("L")
    return any(count for value, count in enumerate(mask.histogram()) if 0 < value < 255 and count)
