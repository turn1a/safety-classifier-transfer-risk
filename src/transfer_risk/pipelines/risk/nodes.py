"""Nodes for the risk pipeline (placeholders — see ``SPEC.md`` §3.1 steps 6-7, §11)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd


def fit_regressors(master_results_table: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Fit the DecisionTree and RandomForest transfer-rate regressors (SPEC.md §3.1 step 6)."""
    raise NotImplementedError


def run_ablation(
    master_results_table: pd.DataFrame,
    selection: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Compare CKA-guided vs random surrogate selection (bootstrap reseeds + paired t-test)."""
    raise NotImplementedError
