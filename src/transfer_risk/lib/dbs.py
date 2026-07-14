"""Diagonal Box Similarity (DBS) over a layer-by-layer CKA matrix.

DBS follows a discrete diagonal from the top-left to bottom-right matrix corner via
Bresenham's line algorithm, then averages cells in the union of in-bounds square boxes
of half-width ``box`` centred on that path. This keeps the original rectangular matrix
intact (no resampling to square), while matching the usual diagonal-band behaviour for
square matrices.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
_MATRIX_NDIM = 2


def _bresenham_diagonal(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    """Return Bresenham cells on the diagonal from ``(0,0)`` to ``(n_rows-1,n_cols-1)``.

    Args:
        n_rows: Number of matrix rows.
        n_cols: Number of matrix columns.

    Returns:
        Ordered ``(row, col)`` coordinates along the discrete diagonal.
    """
    row = 0
    col = 0
    target_row = n_rows - 1
    target_col = n_cols - 1
    row_delta = abs(target_row - row)
    col_delta = abs(target_col - col)
    error = row_delta - col_delta
    path: list[tuple[int, int]] = []
    while True:
        path.append((row, col))
        if row == target_row and col == target_col:
            return path
        doubled_error = 2 * error
        if doubled_error > -col_delta:
            error -= col_delta
            row += 1
        if doubled_error < row_delta:
            error += row_delta
            col += 1


def _diagonal_box_mask(n_rows: int, n_cols: int, box: int) -> npt.NDArray[np.bool_]:
    """Build the DBS mask: union of in-bounds square boxes around a Bresenham diagonal.

    Args:
        n_rows: Number of matrix rows.
        n_cols: Number of matrix columns.
        box: Half-width of each square box around a diagonal cell.

    Returns:
        Boolean mask of selected cells.
    """
    mask = np.zeros((n_rows, n_cols), dtype=np.bool_)
    for row, col in _bresenham_diagonal(n_rows, n_cols):
        row_start = max(0, row - box)
        row_stop = min(n_rows, row + box + 1)
        col_start = max(0, col - box)
        col_stop = min(n_cols, col + box + 1)
        mask[row_start:row_stop, col_start:col_stop] = True
    return mask


def diagonal_box_similarity(matrix: FloatArray, box: int) -> float:
    """Average CKA cells within Bresenham-centred diagonal boxes.

    Args:
        matrix: ``(L_a, L_b)`` layer-by-layer CKA matrix (square or rectangular).
        box: Half-width of each square box around every Bresenham diagonal cell.
            ``box == 0`` averages the strict Bresenham diagonal.

    Returns:
        Mean of the selected cells, in ``[0, 1]``.

    Raises:
        ValueError: If ``matrix`` is not a non-empty 2D matrix, or if ``box`` is negative.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != _MATRIX_NDIM or m.shape[0] == 0 or m.shape[1] == 0:
        msg = "matrix must be a non-empty 2D array"
        raise ValueError(msg)
    if box < 0:
        msg = "box must be non-negative"
        raise ValueError(msg)
    mask = _diagonal_box_mask(m.shape[0], m.shape[1], box)
    return float(m[mask].mean())
