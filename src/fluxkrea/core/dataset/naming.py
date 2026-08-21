"""How a dataset folder becomes an id.

Shared by the daemon's registry and the push client, because both have to
agree: the client asks a node for ``poses`` and the node has to have
called it that. Derived rather than allocated, so the same folder gets the
same id on every node it lives on - which is what makes ``fk dataset
where`` and the drift report readable.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_NOT_SLUG = re.compile(r"[^a-z0-9]+")


def slug(name: str) -> str:
    cleaned = _NOT_SLUG.sub("-", name.strip().lower()).strip("-")
    return cleaned or "dataset"


def derive_id(path: str | Path) -> str:
    """The id a folder gets by default: its own name, slugged."""
    return slug(Path(path).name)


def path_digest(path: str | Path) -> str:
    """Short hash of a path, appended only when two folders want one name."""
    text = Path(path).as_posix().lower()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]


__all__ = ["derive_id", "path_digest", "slug"]
