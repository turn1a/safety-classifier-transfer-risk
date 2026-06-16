"""Nodes for the similarity pipeline (placeholders — see ``SPEC.md`` §3.1-§3.3).

Node bodies will delegate the pure maths to :mod:`transfer_risk.lib.cka`,
:mod:`transfer_risk.lib.dbs`, and :mod:`transfer_risk.lib.thresholds`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd


def build_probe_set(canonical: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Sample a fixed, deterministic probe set of benign + injection prompts."""
    raise NotImplementedError


def compute_cka_matrices(
    probe_set: pd.DataFrame,
    checkpoints: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Capture per-layer hidden states and build target-vs-surrogate CKA matrices."""
    raise NotImplementedError


def reduce_similarity(cka_matrices: dict[str, Any], params: dict[str, Any]) -> pd.DataFrame:
    """Reduce each CKA matrix to (mean CKA, DBS) scalars per surrogate."""
    raise NotImplementedError


def calibrate_thresholds(
    similarity_table: pd.DataFrame, params: dict[str, Any]
) -> dict[str, float]:
    """Calibrate r1/r2 from the observed similarity distribution (quartiles)."""
    raise NotImplementedError


def select_surrogates(
    similarity_table: pd.DataFrame,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Split surrogates into high-similarity (M1) and low-similarity (M2) pools."""
    raise NotImplementedError
