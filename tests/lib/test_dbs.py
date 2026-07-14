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


def test_rectangular_box_zero_follows_bresenham_diagonal() -> None:
    # For a (5, 3) matrix, Bresenham from (0, 0) to (4, 2) visits:
    # (0,0), (1,0), (2,1), (3,1), (4,2). The old normalized-depth resampling
    # path would have used only (0,0), (2,1), (4,2), so this case distinguishes them.
    matrix = np.array(
        [
            [1.0, 0.4, 0.9],
            [0.1, 0.2, 0.3],
            [0.6, 0.9, 0.0],
            [0.8, 0.2, 0.5],
            [0.7, 0.6, 0.8],
        ]
    )
    assert diagonal_box_similarity(matrix, 0) == pytest.approx(0.6)


def test_rectangular_box_uses_union_of_clipped_squares() -> None:
    matrix = np.arange(15.0, dtype=np.float64).reshape(5, 3)
    # With the same (5, 3) Bresenham path as above and box=1, the union of in-bounds
    # squares includes every cell except (0, 2).
    expected = float((matrix.sum() - matrix[0, 2]) / 14.0)
    assert diagonal_box_similarity(matrix, 1) == pytest.approx(expected)


def test_rectangular_wide_box_covers_full_matrix() -> None:
    matrix = np.arange(15.0, dtype=np.float64).reshape(5, 3)
    assert diagonal_box_similarity(matrix, 100) == pytest.approx(float(matrix.mean()))


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([0.1, 0.2]),
        np.ones((1, 1, 1)),
        np.empty((0, 2)),
        np.empty((2, 0)),
    ],
)
def test_diagonal_box_similarity_requires_nonempty_two_dimensional_matrix(
    matrix: np.ndarray,
) -> None:
    """DBS rejects scalar-like, higher-rank, and empty matrix inputs."""
    with pytest.raises(ValueError, match="non-empty 2D array"):
        diagonal_box_similarity(matrix, box=0)


def test_diagonal_box_similarity_rejects_negative_box_half_width() -> None:
    """A diagonal box cannot have a negative geometric half-width."""
    with pytest.raises(ValueError, match="box must be non-negative"):
        diagonal_box_similarity(np.eye(2), box=-1)
