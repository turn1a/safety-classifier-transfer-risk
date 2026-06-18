"""CKA invariants and the minibatch cross-check (SPEC.md §12)."""

from __future__ import annotations

import numpy as np
import pytest

from transfer_risk.lib.cka import cka_matrix, linear_cka, minibatch_cka


def test_self_similarity_is_one() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((48, 16))
    assert linear_cka(x, x) == pytest.approx(1.0)


def test_invariant_to_orthogonal_rotation() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((48, 16))
    q, _ = np.linalg.qr(rng.standard_normal((16, 16)))
    assert linear_cka(x, x @ q) == pytest.approx(1.0)


def test_invariant_to_isotropic_scaling() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((48, 16))
    assert linear_cka(x, 3.5 * x) == pytest.approx(1.0)


def test_distinct_representations_below_one() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((64, 16))
    y = rng.standard_normal((64, 16))
    assert 0.0 <= linear_cka(x, y) < 1.0


def test_minibatch_matches_full_batch() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((256, 32))
    y = x @ rng.standard_normal((32, 24)) + 0.1 * rng.standard_normal((256, 24))
    full = linear_cka(x, y)
    batched = minibatch_cka(np.array_split(x, 4), np.array_split(y, 4))
    assert batched == pytest.approx(full, abs=0.06)


def test_constant_representation_is_zero_not_nan() -> None:
    rng = np.random.default_rng(6)
    x = rng.standard_normal((32, 8))
    assert linear_cka(np.ones((32, 8)), x) == 0.0


def test_cka_matrix_diagonal_is_one_for_identical_layer_lists() -> None:
    rng = np.random.default_rng(5)
    reps = [rng.standard_normal((40, 12)), rng.standard_normal((40, 8))]
    matrix = cka_matrix(reps, reps)
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[1, 1] == pytest.approx(1.0)
    assert matrix[0, 1] < 1.0
