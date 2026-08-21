"""Analytics computed above the backend line.

Backends emit events; this package derives everything else from them, so
every backend gets the same features rather than whichever ones its author
happened to implement (doc 02).
"""

from .loss import EMA_WINDOWS, LossSeries, Outlier, Trend

__all__ = ["EMA_WINDOWS", "LossSeries", "Outlier", "Trend"]
