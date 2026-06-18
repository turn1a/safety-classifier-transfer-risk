"""Selection ablation: high-CKA (M1) vs low-CKA (M2) surrogates (SPEC.md §11).

Pure helpers (no I/O) for the risk stage's selection ablation. The headline question is
whether choosing surrogates by representational similarity helps an attacker: do the
high-similarity surrogates (M1) yield more transferable attacks than the low-similarity
surrogates (M2)? :func:`selection_ablation` answers it with a one-sided permutation test on
the difference in group-mean transfer rate, evaluated on both the per-surrogate
mean-across-recipes and max-across-recipes summaries. The node in ``pipelines/risk`` supplies
the per-surrogate transfer summaries, the M1/M2 membership, and a seeded generator.

The test permutes the M1/M2 labels over the combined ``M1 + M2`` pool and recomputes the
difference in group means to build the null distribution. When the number of distinct label
assignments ``C(n, |M1|)`` is small it is enumerated exactly (an exact permutation p-value);
otherwise it is sampled (Monte Carlo). Small groups give a coarse p-value floor — three vs
three surrogates have ``C(6, 3) = 20`` assignments, so the smallest one-sided p-value is
``1 / 20 = 0.05`` — which the risk node and the report state explicitly.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]

# Above this many distinct label assignments, sample the null instead of enumerating it.
_EXACT_ENUMERATION_CAP = 2000


class PermutationResult(TypedDict):
    """Outcome of a one-sided two-group permutation test on the difference in means.

    Attributes:
        observed_diff: The observed ``mean(A) - mean(B)``.
        p_value: Fraction of label assignments at least as extreme as the observed one.
        null_mean: Mean of the permutation null distribution.
        null_std: Standard deviation of the permutation null distribution.
        exact: Whether the null was enumerated exactly (vs sampled).
        n_used: Number of assignments considered, including the observed one.
    """

    observed_diff: float
    p_value: float
    null_mean: float
    null_std: float
    exact: bool
    n_used: int


class SelectionAblation(TypedDict):
    """M1-vs-M2 selection-ablation result on mean and max transfer rate.

    Attributes:
        m1_mean: Mean over M1 of each surrogate's mean transfer rate.
        m2_mean: Mean over M2 of each surrogate's mean transfer rate.
        mean_diff_pp: ``m1_mean - m2_mean`` in percentage points.
        mean_p_value: One-sided permutation p-value for the mean-transfer contrast.
        m1_max_mean: Mean over M1 of each surrogate's max transfer rate.
        m2_max_mean: Mean over M2 of each surrogate's max transfer rate.
        max_diff_pp: ``m1_max_mean - m2_max_mean`` in percentage points.
        max_p_value: One-sided permutation p-value for the max-transfer contrast.
        exact: Whether the permutation nulls were enumerated exactly.
        n_m1: Number of surrogates in M1.
        n_m2: Number of surrogates in M2.
    """

    m1_mean: float
    m2_mean: float
    mean_diff_pp: float
    mean_p_value: float
    m1_max_mean: float
    m2_max_mean: float
    max_diff_pp: float
    max_p_value: float
    exact: bool
    n_m1: int
    n_m2: int


def _mask_from_indices(indices: Iterable[int], n: int) -> BoolArray:
    """Build a length-``n`` boolean mask that is ``True`` at ``indices``."""
    mask: BoolArray = np.zeros(n, dtype=bool)
    mask[list(indices)] = True
    return mask


def _mean_difference(values: FloatArray, in_group_a: BoolArray) -> float:
    """Difference in group means: ``mean(values[A]) - mean(values[~A])``."""
    return float(values[in_group_a].mean() - values[~in_group_a].mean())


def permutation_test(
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    n_permutations: int,
    rng: np.random.Generator,
    alternative: str = "greater",
) -> PermutationResult:
    """Run a one-sided permutation test on the difference in group means.

    Pools ``group_a`` and ``group_b``, repeatedly reassigns the group labels at the observed
    group sizes, and recomputes ``mean(A) - mean(B)`` to build the null distribution. The
    p-value is the fraction of assignments whose statistic is at least as extreme as the
    observed one in the ``alternative`` direction, counting the observed assignment itself.
    When the number of distinct assignments ``C(n, |A|)`` does not exceed
    :data:`_EXACT_ENUMERATION_CAP` every assignment is enumerated (an exact p-value);
    otherwise ``n_permutations`` random assignments are sampled and the p-value uses the
    ``(hits + 1) / (samples + 1)`` correction so it is never zero.

    Args:
        group_a: Per-item scalars for group A (here, M1's per-surrogate transfer summary).
        group_b: Per-item scalars for group B (here, M2's).
        n_permutations: Number of random assignments when sampling (ignored when exact).
        rng: A seeded NumPy generator (reproducibility).
        alternative: ``"greater"`` tests ``A > B``; ``"less"`` tests ``A < B``.

    Returns:
        A :class:`PermutationResult`.

    Raises:
        ValueError: If either group is empty or ``alternative`` is unrecognised.
    """
    if alternative not in {"greater", "less"}:
        msg = f"alternative must be 'greater' or 'less', got {alternative!r}"
        raise ValueError(msg)
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        msg = "both groups must be non-empty"
        raise ValueError(msg)
    values: FloatArray = np.concatenate([a, b])
    n = int(values.size)
    size_a = int(a.size)
    observed = _mean_difference(values, _mask_from_indices(range(size_a), n))

    exact = comb(n, size_a) <= _EXACT_ENUMERATION_CAP
    if exact:
        null = np.array(
            [
                _mean_difference(values, _mask_from_indices(indices, n))
                for indices in combinations(range(n), size_a)
            ]
        )
    else:
        null = np.array(
            [
                _mean_difference(
                    values, _mask_from_indices(rng.choice(n, size_a, replace=False), n)
                )
                for _ in range(n_permutations)
            ]
        )

    extreme = null >= observed if alternative == "greater" else null <= observed
    hits = int(extreme.sum())
    if exact:
        # The observed assignment is one of the enumerated combinations, so it is already
        # counted; the exact p-value is the fraction of assignments at least as extreme.
        p_value = hits / null.size
        n_used = int(null.size)
    else:
        # The observed assignment is not guaranteed to be sampled; add it to numerator and
        # denominator so the p-value is never zero (it is at least as extreme as itself).
        p_value = (hits + 1) / (null.size + 1)
        n_used = int(null.size + 1)
    return PermutationResult(
        observed_diff=observed,
        p_value=float(p_value),
        null_mean=float(null.mean()),
        null_std=float(null.std()),
        exact=bool(exact),
        n_used=n_used,
    )


def selection_ablation(
    transfer_mean: Mapping[str, float],
    transfer_max: Mapping[str, float],
    m1: Sequence[str],
    m2: Sequence[str],
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> SelectionAblation:
    """Compare high-CKA (M1) and low-CKA (M2) surrogates on transfer rate.

    Runs :func:`permutation_test` twice — once on each surrogate's mean transfer rate across
    recipes, once on its max — testing the one-sided hypothesis that M1 transfers more than
    M2. Group means are reported both raw and as a percentage-point difference.

    Args:
        transfer_mean: Per-surrogate mean transfer rate (averaged over the surrogate's
            recipes).
        transfer_max: Per-surrogate max transfer rate (over the surrogate's recipes).
        m1: High-similarity surrogate names (CKA >= r1).
        m2: Low-similarity surrogate names (CKA <= r2).
        n_permutations: Random assignments for the sampled path (see :func:`permutation_test`).
        rng: A seeded NumPy generator (reproducibility).

    Returns:
        A :class:`SelectionAblation`.

    Raises:
        ValueError: If either group is empty (no surrogate to contrast).
    """
    if not m1 or not m2:
        msg = "selection_ablation needs at least one surrogate in each of M1 and M2"
        raise ValueError(msg)
    mean_a = [transfer_mean[name] for name in m1]
    mean_b = [transfer_mean[name] for name in m2]
    max_a = [transfer_max[name] for name in m1]
    max_b = [transfer_max[name] for name in m2]
    mean_test = permutation_test(
        mean_a, mean_b, n_permutations=n_permutations, rng=rng, alternative="greater"
    )
    max_test = permutation_test(
        max_a, max_b, n_permutations=n_permutations, rng=rng, alternative="greater"
    )
    return SelectionAblation(
        m1_mean=float(np.mean(mean_a)),
        m2_mean=float(np.mean(mean_b)),
        mean_diff_pp=mean_test["observed_diff"] * 100.0,
        mean_p_value=mean_test["p_value"],
        m1_max_mean=float(np.mean(max_a)),
        m2_max_mean=float(np.mean(max_b)),
        max_diff_pp=max_test["observed_diff"] * 100.0,
        max_p_value=max_test["p_value"],
        exact=bool(mean_test["exact"] and max_test["exact"]),
        n_m1=len(m1),
        n_m2=len(m2),
    )
