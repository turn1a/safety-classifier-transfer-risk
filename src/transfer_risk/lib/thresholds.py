"""Empirical calibration of the surrogate-selection thresholds r1 and r2.

The Cox & Bunzel method splits the surrogate pool by representational similarity:
``M1`` (high, similarity >= r1) and ``M2`` (low, similarity <= r2). Thresholds are
calibrated from the *observed* similarity distribution (upper/lower quartiles by
default), never copied from the paper's CNN-derived values (SPEC.md §3.1 step 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class Thresholds:
    """Similarity thresholds bracketing the surrogate pool.

    Attributes:
        r1: High-similarity threshold (upper quartile); surrogates >= r1 form M1.
        r2: Low-similarity threshold (lower quartile); surrogates <= r2 form M2.
    """

    r1: float
    r2: float


def calibrate(
    similarities: Sequence[float],
    r1_quantile: float = 0.75,
    r2_quantile: float = 0.25,
) -> Thresholds:
    """Calibrate r1/r2 from an observed similarity distribution.

    Args:
        similarities: Pairwise target-vs-surrogate similarity scores.
        r1_quantile: Upper quantile for the high-similarity cut (default 0.75).
        r2_quantile: Lower quantile for the low-similarity cut (default 0.25).

    Returns:
        :class:`Thresholds` satisfying ``0 < r2 < r1 < 1``.

    Raises:
        ValueError: If the distribution is empty or degenerate (no ``0 < r2 < r1 < 1``).
    """
    arr = np.asarray(list(similarities), dtype=np.float64)
    if arr.size == 0:
        msg = "cannot calibrate thresholds from an empty similarity distribution"
        raise ValueError(msg)
    r1 = float(np.quantile(arr, r1_quantile))
    r2 = float(np.quantile(arr, r2_quantile))
    if not 0.0 < r2 < r1 < 1.0:
        msg = f"degenerate thresholds: require 0 < r2 < r1 < 1, got r2={r2}, r1={r1}"
        raise ValueError(msg)
    return Thresholds(r1=r1, r2=r2)
