"""Tests for transfer_risk.lib.ablation: the M1-vs-M2 selection ablation (SPEC.md §11).

The invariants are behavioural, not smoke. The exact permutation p-value of a hand-computable
split is checked against the closed form; the test is directional (M1 > M2 is significant, the
reverse is not); the sampled path is reproducible under a fixed seed and recovers a small
p-value for a cleanly separated split; and the public contract (empty groups, bad alternative)
raises.
"""

import numpy as np
import pytest

from transfer_risk.lib.ablation import permutation_test, selection_ablation


def test_exact_pvalue_matches_closed_form() -> None:
    # M1 = {0.9, 0.8}, M2 = {0.1, 0.2}: of the C(4,2)=6 label assignments only the observed
    # one reaches the 0.70 mean difference, so the exact one-sided p-value is 1/6.
    result = permutation_test(
        [0.9, 0.8], [0.1, 0.2], n_permutations=0, rng=np.random.default_rng(0)
    )
    assert result["exact"] is True
    assert result["n_used"] == 6
    assert result["observed_diff"] == pytest.approx(0.70)
    assert result["p_value"] == pytest.approx(1.0 / 6.0)


def test_three_vs_three_has_a_floor_of_one_in_twenty() -> None:
    # The most extreme split of 3 vs 3 reaches the smallest exact one-sided p-value, 1/20.
    result = permutation_test(
        [0.9, 0.8, 0.7], [0.3, 0.2, 0.1], n_permutations=0, rng=np.random.default_rng(0)
    )
    assert result["exact"] is True
    assert result["n_used"] == 20
    assert result["p_value"] == pytest.approx(1.0 / 20.0)


def test_reverse_direction_is_not_significant() -> None:
    # Swapping the groups makes the observed difference the least extreme, so every
    # assignment is at least as large and the one-sided p-value is 1.0.
    result = permutation_test(
        [0.1, 0.2], [0.9, 0.8], n_permutations=0, rng=np.random.default_rng(0)
    )
    assert result["observed_diff"] == pytest.approx(-0.70)
    assert result["p_value"] == pytest.approx(1.0)


def test_alternative_less_mirrors_greater() -> None:
    greater = permutation_test(
        [0.1, 0.2], [0.9, 0.8], n_permutations=0, rng=np.random.default_rng(0), alternative="less"
    )
    assert greater["p_value"] == pytest.approx(1.0 / 6.0)


def test_sampled_path_is_reproducible_and_small_for_separated_groups() -> None:
    # 10 vs 10 has C(20,10) = 184_756 > the enumeration cap, so the null is sampled.
    high = [0.9] * 10
    low = [0.1] * 10
    first = permutation_test(high, low, n_permutations=999, rng=np.random.default_rng(7))
    second = permutation_test(high, low, n_permutations=999, rng=np.random.default_rng(7))
    assert first["exact"] is False
    assert first["p_value"] == second["p_value"]  # same seed -> identical
    # The true split is the most extreme; a random draw almost never reproduces it.
    assert first["p_value"] < 0.01


def test_permutation_test_rejects_empty_group() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        permutation_test([], [0.1], n_permutations=10, rng=np.random.default_rng(0))


def test_permutation_test_rejects_unknown_alternative() -> None:
    with pytest.raises(ValueError, match="alternative"):
        permutation_test(
            [0.9], [0.1], n_permutations=10, rng=np.random.default_rng(0), alternative="two-sided"
        )


def test_selection_ablation_reports_both_contrasts() -> None:
    transfer_mean = {"hi1": 0.6, "hi2": 0.5, "lo1": 0.1, "lo2": 0.2}
    transfer_max = {"hi1": 0.8, "hi2": 0.7, "lo1": 0.2, "lo2": 0.3}
    result = selection_ablation(
        transfer_mean,
        transfer_max,
        m1=["hi1", "hi2"],
        m2=["lo1", "lo2"],
        n_permutations=0,
        rng=np.random.default_rng(0),
    )
    assert result["m1_mean"] == pytest.approx(0.55)
    assert result["m2_mean"] == pytest.approx(0.15)
    assert result["mean_diff_pp"] == pytest.approx(40.0)
    assert result["max_diff_pp"] == pytest.approx(50.0)
    assert result["mean_p_value"] == pytest.approx(1.0 / 6.0)
    assert result["max_p_value"] == pytest.approx(1.0 / 6.0)
    assert result["exact"] is True
    assert result["n_m1"] == 2
    assert result["n_m2"] == 2


def test_selection_ablation_requires_both_groups() -> None:
    with pytest.raises(ValueError, match="M1 and M2"):
        selection_ablation(
            {"a": 0.5}, {"a": 0.5}, m1=["a"], m2=[], n_permutations=0, rng=np.random.default_rng(0)
        )
