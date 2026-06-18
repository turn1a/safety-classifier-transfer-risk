"""Diagonal Box Similarity (DBS) over a layer-by-layer CKA matrix.

DBS averages the CKA cells within a band of half-width ``box`` around the diagonal
(SPEC.md §3.3): ``box == 0`` is the strict-diagonal mean and ``box >= L`` the full-matrix
mean. For a rectangular matrix (models of different depth — e.g. a 13-layer encoder vs
the 2-layer BiLSTM) the diagonal is undefined, so the larger axis is first aligned onto
the smaller by sampling normalised depth, then the resulting square is banded.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _align_to_square(matrix: FloatArray) -> FloatArray:
    """Resample a rectangular matrix to ``(k, k)`` by normalised-depth sampling."""
    n_rows, n_cols = matrix.shape
    k = min(n_rows, n_cols)
    rows = np.linspace(0, n_rows - 1, k).round().astype(int)
    cols = np.linspace(0, n_cols - 1, k).round().astype(int)
    aligned: FloatArray = matrix[np.ix_(rows, cols)]
    return aligned


def diagonal_box_similarity(matrix: FloatArray, box: int) -> float:
    """Average CKA cells within a diagonal box of half-width ``box``.

    Args:
        matrix: ``(L_a, L_b)`` layer-by-layer CKA matrix (square or rectangular).
        box: Half-width of the diagonal box (``0`` selects the strict diagonal).

    Returns:
        Mean of the selected cells, in ``[0, 1]``.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape[0] != m.shape[1]:
        m = _align_to_square(m)
    side = m.shape[0]
    rows, cols = np.indices((side, side))
    mask = np.abs(rows - cols) <= box
    return float(m[mask].mean())
