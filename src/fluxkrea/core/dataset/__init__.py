"""The dataset model and every operation that touches a dataset folder.

Nothing outside this package opens a folder and globs for images. One
scanner, one extension list, one definition of what a training example is
(doc 03).
"""

from .item import DatasetItem
from .metadata import Metadata
from .scan import scan
from .validate import Problem, ValidationReport, validate

__all__ = [
    "DatasetItem",
    "Metadata",
    "Problem",
    "ValidationReport",
    "scan",
    "validate",
]
