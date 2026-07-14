"""Tests for surrogate-level Spearman association helpers."""

from __future__ import annotations

import math

import pytest

from transfer_risk.lib.association import _tie_free_abs_rho_null_distribution, spearman_association


def test_spearman_association_matches_saved_data_headline() -> None:
    mean_cka = [
        0.25871896606466466,
        0.329011060483958,
        0.4123087022398452,
        0.4169064105029701,
        0.4284723991738573,
        0.4295596540530011,
        0.4450094194724594,
        0.451372545278504,
        0.4542644298102016,
        0.4754217532198447,
    ]
    mean_transfer = [
        0.1752838286577085,
        0.34539311694480757,
        0.051050072408211945,
        0.17711537358247426,
        0.5416740988480119,
        0.5443940664974856,
        0.4449768967416026,
        0.33792998550618536,
        0.556989247311828,
        0.6468421052631579,
    ]
    stats = spearman_association(mean_cka, mean_transfer)
    assert stats["rho"] == pytest.approx(0.7575758)
    assert stats["two_sided_p"] == pytest.approx(0.01492945)
    assert stats["n"] == 10
    assert stats["exact"] is True


def test_spearman_association_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        spearman_association([0.1, 0.2], [0.1])


def test_spearman_association_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        spearman_association([0.1], [0.2])


def test_spearman_association_rejects_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        spearman_association([0.1, float("nan")], [0.2, 0.3])


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([1.0, 1.0, 1.0], [0.1, 0.2, 0.3]),
        ([0.1, 0.2, 0.3], [1.0, 1.0, 1.0]),
    ],
)
def test_spearman_association_rejects_constant_sequence(
    first: list[float],
    second: list[float],
) -> None:
    with pytest.raises(ValueError, match="undefined for constant input sequences"):
        spearman_association(first, second)


@pytest.mark.parametrize("enumeration_cap", [0, -1, -10])
def test_spearman_association_rejects_nonpositive_enumeration_cap(enumeration_cap: int) -> None:
    with pytest.raises(ValueError, match="enumeration_cap must be positive"):
        spearman_association([0.1, 0.2], [0.2, 0.3], enumeration_cap=enumeration_cap)


def test_tie_free_exact_null_distribution_is_cached_by_n() -> None:
    _tie_free_abs_rho_null_distribution.cache_clear()
    first = _tie_free_abs_rho_null_distribution(10)
    second = _tie_free_abs_rho_null_distribution(10)
    third = _tie_free_abs_rho_null_distribution(9)
    assert first is second
    assert sum(count for _, count in first) == math.factorial(10)
    assert sum(count for _, count in third) == math.factorial(9)
