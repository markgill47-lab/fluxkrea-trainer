"""Shared fixtures.

Two things matter here. First, no test ever touches the real user profile:
``isolated_env`` is autouse and points every location in ``core.paths`` at
a temp directory. Second, dataset fixtures build real files on disk -
the v1 bugs this project exists to prevent were all file-level, and a
mocked filesystem would have hidden every one of them.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fluxkrea.core import paths

#: Every environment variable the app reads, cleared before each test so a
#: developer's own shell cannot change a result.
APP_ENV_VARS = (
    "FLUXKREA_HOME",
    "FLUXKREA_CONFIG_DIR",
    "FLUXKREA_CONFIG_FILE",
    "FLUXKREA_DATA_DIR",
    "FLUXKREA_CACHE_DIR",
    "FLUXKREA_STATE_DIR",
    "FLUXKREA_FLEET_FILE",
    "FLUXKREA_ASSETS_DIR",
    "FLUXKREA_TOKEN",
    "FLUXKREA_CLAUDE_API_KEY",
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY",
    "FLUXKREA_LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Relocate config, data, cache and state under a temp directory."""
    for name in APP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith("FLUXKREA_"):
            monkeypatch.delenv(name, raising=False)

    root = tmp_path / "profile"
    monkeypatch.setenv("FLUXKREA_HOME", str(root))
    return root


# --------------------------------------------------------------------------
# image and dataset builders
# --------------------------------------------------------------------------


def make_image(
    path: Path,
    size: tuple[int, int] = (64, 48),
    colour: tuple[int, int, int] = (120, 90, 60),
    *,
    exif_orientation: int | None = None,
) -> Path:
    """Write a real image file. Deterministic content, so digests are stable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, colour)
    # A gradient corner, so a flip or a rotate is detectable by inspection.
    pixels = image.load()
    for y in range(min(8, size[1])):
        for x in range(min(8, size[0])):
            pixels[x, y] = (255, 255, 255)

    kwargs: dict[str, object] = {}
    if exif_orientation is not None:
        exif = image.getexif()
        exif[274] = exif_orientation  # 274 == Orientation
        kwargs["exif"] = exif
    image.save(path, **kwargs)
    return path


def make_mask(path: Path, size: tuple[int, int] = (64, 48), box: tuple[int, int, int, int] | None = None) -> Path:
    """Write an 8-bit greyscale mask: white background, black box.

    Faces black, everything else white - white is weight 1, i.e. trained
    (doc 04, "contract").
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((size[1], size[0]), 255, dtype=np.uint8)
    if box:
        x, y, w, h = box
        array[y : y + h, x : x + w] = 0
    Image.fromarray(array, mode="L").save(path)
    return path


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    return tmp_path / "poses"


@pytest.fixture
def dataset(dataset_dir: Path) -> Path:
    """A small, well-formed dataset: four bundles, one of them masked."""
    for index in range(1, 5):
        stem = f"punch_{index:03d}"
        make_image(dataset_dir / f"{stem}.jpg", size=(64 + index, 48 + index))
        (dataset_dir / f"{stem}.txt").write_text(f"a fighter throwing punch {index}", encoding="utf-8")
    make_mask(
        paths.masks_dir(dataset_dir) / "punch_001.png",
        size=(65, 49),
        box=(10, 10, 20, 20),
    )
    return dataset_dir


@pytest.fixture
def collector() -> Iterator:
    from fluxkrea.core.events import Collector

    yield Collector()
