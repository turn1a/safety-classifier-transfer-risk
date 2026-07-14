"""Pure surrogate-level association helpers for risk evidence.

This module computes Spearman association between two aligned numeric sequences and
reports an exact two-sided permutation p-value when exhaustive enumeration is tractable
(by default up to ``10!`` permutations). Above that cap, it falls back to a deterministic
normal approximation to avoid impractical brute force.
"""

from __future__ import annotations

import itertools
import math
from functools import cache

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
_DEFAULT_ENUMERATION_CAP = math.factorial(10)
_EPSILON = 1e-12
_MIN_OBSERVATIONS = 2


def _as_finite_array(values: npt.ArrayLike, *, name: str) -> FloatArray:
    """Convert a numeric sequence to a finite float array.

    Args:
        values: Input numeric sequence.
        name: Human-readable argument name for error messages.

    Returns:
        The values as a one-dimensional float64 NumPy array.

    Raises:
        ValueError: If values are not one-dimensional or contain non-finite numbers.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        msg = f"{name} must be a one-dimensional sequence"
        raise ValueError(msg)
    if not np.isfinite(array).all():
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return array


def _average_ranks(values: FloatArray) -> FloatArray:
    """Assign average ranks (1-based) with tie handling.

    Args:
        values: Input values.

    Returns:
        Rank array with average ranks for ties.
    """
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_values.shape[0]:
        stop = start + 1
        while stop < sorted_values.shape[0] and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = (start + stop - 1) / 2.0 + 1.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def _pearson_correlation(x: FloatArray, y: FloatArray) -> float:
    """Compute Pearson correlation for two aligned arrays.

    Args:
        x: First numeric vector.
        y: Second numeric vector.

    Returns:
        Pearson correlation coefficient.

    Raises:
        ValueError: If either vector is constant and the correlation is undefined.
    """
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= 0.0:
        msg = "Spearman association is undefined for constant input sequences"
        raise ValueError(msg)
    numerator = float(np.dot(x_centered, y_centered))
    return numerator / denominator


@cache
def _tie_free_abs_rho_null_distribution(n: int) -> tuple[tuple[float, int], ...]:
    """Enumerate and cache tie-free absolute-rho null mass for a sample size.

    Args:
        n: Number of observations.

    Returns:
        Immutable sorted ``(abs_rho, count)`` pairs for all ``n!`` permutations.
    """
    denominator = n * (n * n - 1)
    factor = 6.0 / float(denominator)
    counts: dict[float, int] = {}
    for permutation in itertools.permutations(range(n)):
        distance = 0
        for index, permuted_index in enumerate(permutation):
            diff = index - permuted_index
            distance += diff * diff
        abs_rho = abs(1.0 - factor * float(distance))
        counts[abs_rho] = counts.get(abs_rho, 0) + 1
    return tuple(sorted(counts.items(), key=lambda item: item[0]))


def _exact_two_sided_pvalue(observed_rho: float, n: int) -> float:
    """Compute the exact two-sided permutation p-value for tie-free ranks.

    Args:
        observed_rho: Observed Spearman rho.
        n: Number of observations.

    Returns:
        Exact two-sided p-value under the null of permutation independence.
    """
    observed_abs = abs(observed_rho)
    extreme_count = 0
    threshold = observed_abs - _EPSILON
    for abs_rho, count in _tie_free_abs_rho_null_distribution(n):
        if abs_rho >= threshold:
            extreme_count += count
    return float(extreme_count / math.factorial(n))


def _normal_approx_two_sided_pvalue(observed_rho: float, n: int) -> float:
    """Approximate a two-sided p-value for Spearman rho via normal asymptotics.

    Args:
        observed_rho: Observed Spearman rho.
        n: Number of observations.

    Returns:
        Approximate two-sided p-value in ``[0, 1]``.
    """
    z = abs(observed_rho) * math.sqrt(float(n - 1))
    p_value = math.erfc(z / math.sqrt(2.0))
    return float(min(1.0, max(0.0, p_value)))


def spearman_association(
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    *,
    enumeration_cap: int = _DEFAULT_ENUMERATION_CAP,
) -> dict[str, float | int | bool]:
    """Compute Spearman association for aligned numeric sequences.

    Args:
        first: First numeric sequence.
        second: Second numeric sequence, aligned with ``first``.
        enumeration_cap: Maximum number of permutations to enumerate exactly. If
            ``n!`` exceeds this cap, p-value computation falls back to a deterministic
            normal approximation.

    Returns:
        Dict with ``rho``, ``two_sided_p``, ``n``, and ``exact``.

    Raises:
        ValueError: If lengths mismatch, fewer than two observations are provided,
            values are non-finite, or ``enumeration_cap`` is not positive.
    """
    if enumeration_cap <= 0:
        msg = "enumeration_cap must be positive"
        raise ValueError(msg)
    x = _as_finite_array(first, name="first")
    y = _as_finite_array(second, name="second")
    if x.shape[0] != y.shape[0]:
        msg = "first and second must have the same length"
        raise ValueError(msg)
    n = int(x.shape[0])
    if n < _MIN_OBSERVATIONS:
        msg = f"Spearman association needs at least {_MIN_OBSERVATIONS} observations"
        raise ValueError(msg)
    x_ranks = _average_ranks(x)
    y_ranks = _average_ranks(y)
    rho = _pearson_correlation(x_ranks, y_ranks)
    no_ties = len(np.unique(x)) == n and len(np.unique(y)) == n
    total_permutations = math.factorial(n)
    exact = no_ties and total_permutations <= enumeration_cap
    p_value = _exact_two_sided_pvalue(rho, n) if exact else _normal_approx_two_sided_pvalue(rho, n)
    return {"rho": rho, "two_sided_p": p_value, "n": n, "exact": exact}
