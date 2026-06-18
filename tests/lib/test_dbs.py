"""Diagonal Box Similarity edge cases (SPEC.md §3.3, §12)."""

from __future__ import annotations

import numpy as np
import pytest

from transfer_risk.lib.dbs import diagonal_box_similarity


def test_box_zero_is_strict_diagonal_mean() -> None:
    matrix = np.array([[1.0, 0.2], [0.3, 0.8]])
    assert diagonal_box_similarity(matrix, 0) == pytest.approx(0.9)


def test_box_full_is_full_matrix_mean() -> None:
    matrix = np.array([[1.0, 0.2], [0.3, 0.8]])
    assert diagonal_box_similarity(matrix, matrix.shape[0]) == pytest.approx(0.575)


def test_rectangular_aligns_by_normalized_depth() -> None:
    # (4, 2) aligns to (2, 2) by sampling rows [0, 3] and cols [0, 1];
    # box=0 averages the aligned diagonal [m[0, 0]=1.0, m[3, 1]=0.7] -> 0.85.
    matrix = np.array([[1.0, 0.1], [0.9, 0.2], [0.8, 0.3], [0.4, 0.7]])
    assert diagonal_box_similarity(matrix, 0) == pytest.approx(0.85)
