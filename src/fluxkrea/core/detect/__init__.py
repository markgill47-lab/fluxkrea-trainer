"""Face detection, behind an interface (doc 04)."""

from .base import MANUAL, Box, Detector, DetectorError, NullDetector, available, get_detector

__all__ = [
    "MANUAL",
    "Box",
    "Detector",
    "DetectorError",
    "NullDetector",
    "available",
    "get_detector",
]
