"""Linear Centered Kernel Alignment (CKA) for representational similarity.

Linear CKA (Kornblith et al. 2019) measures how similar two sets of activations are,
invariant to orthogonal rotation and isotropic scaling, in ``[0, 1]``. We use the
cross-covariance form ``||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)``
(``_c`` = column-centred), which equals the HSIC/Gram formulation but only ever forms
the ``p x q`` feature matrices, never the ``n x n`` Gram matrix — so it stays exact and
cheap for our probe sizes and needs no minibatching for memory. ``minibatch_cka``
accumulates the same statistic across batches (Nguyen, Raghu & Kornblith 2020) and is
kept as a streaming option and a correctness cross-check.

Reference: SPEC.md §3.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

FloatArray = npt.NDArray[np.float64]


def _centered(a: FloatArray) -> FloatArray:
    """Column-centre an ``(n, d)`` activation matrix in float64."""
    a = np.asarray(a, dtype=np.float64)
    centered: FloatArray = a - a.mean(axis=0, keepdims=True)
    return centered


def _linear_hsic(x_centered: FloatArray, y_centered: FloatArray) -> float:
    """Biased linear HSIC of two already-centred matrices: ``||Y^T X||_F^2``."""
    cross = y_centered.T @ x_centered
    return float(np.sum(cross * cross))


def linear_cka(x: FloatArray, y: FloatArray) -> float:
    """Compute linear CKA between two activation matrices.

    Args:
        x: Activations from model A, shape ``(n_examples, p_features)``.
        y: Activations from model B, shape ``(n_examples, q_features)``.

    Returns:
        CKA similarity in ``[0, 1]``; ``1.0`` for identical representations
        (up to orthogonal rotation and isotropic scaling).
    """
    x_c = _centered(x)
    y_c = _centered(y)
    hsic_xy = _linear_hsic(x_c, y_c)
    hsic_xx = _linear_hsic(x_c, x_c)
    hsic_yy = _linear_hsic(y_c, y_c)
    denominator = np.sqrt(hsic_xx * hsic_yy)
    # A constant (zero-variance) representation has no structure to align with, so
    # define its CKA as 0 rather than 0/0 = NaN — e.g. a transformer's static CLS
    # embedding layer under CLS pooling.
    if denominator == 0.0:
        return 0.0
    return float(hsic_xy / denominator)


def minibatch_cka(x_batches: Iterable[FloatArray], y_batches: Iterable[FloatArray]) -> float:
    """Estimate linear CKA by accumulating HSIC over aligned minibatches.

    Args:
        x_batches: Per-batch activations from model A.
        y_batches: Per-batch activations from model B, aligned with ``x_batches``.

    Returns:
        CKA similarity in ``[0, 1]``.
    """
    sum_xy = sum_xx = sum_yy = 0.0
    for x_b, y_b in zip(x_batches, y_batches, strict=True):
        x_c = _centered(x_b)
        y_c = _centered(y_b)
        sum_xy += _linear_hsic(x_c, y_c)
        sum_xx += _linear_hsic(x_c, x_c)
        sum_yy += _linear_hsic(y_c, y_c)
    denominator = np.sqrt(sum_xx * sum_yy)
    if denominator == 0.0:
        return 0.0
    return float(sum_xy / denominator)


def cka_matrix(reps_a: Sequence[FloatArray], reps_b: Sequence[FloatArray]) -> FloatArray:
    """Build the layer-by-layer CKA matrix between two models' pooled representations.

    Args:
        reps_a: Per-layer ``(n_examples, d)`` activation matrices for model A.
        reps_b: Per-layer activation matrices for model B (same ``n_examples``).

    Returns:
        A ``(len(reps_a), len(reps_b))`` matrix of pairwise layer CKA values.
    """
    matrix: FloatArray = np.empty((len(reps_a), len(reps_b)), dtype=np.float64)
    for i, layer_a in enumerate(reps_a):
        for j, layer_b in enumerate(reps_b):
            matrix[i, j] = linear_cka(layer_a, layer_b)
    return matrix
