"""Nodes for the reporting pipeline (placeholders — see ``SPEC.md`` §10).

Each node returns a Matplotlib ``Figure`` that the catalog persists via
``matplotlib.MatplotlibWriter``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd
    from matplotlib.figure import Figure


def plot_cka_heatmap(similarity_table: pd.DataFrame) -> Figure:
    """Render the target-vs-surrogate CKA similarity heatmap."""
    raise NotImplementedError


def plot_transfer_scatter(master_results_table: pd.DataFrame) -> Figure:
    """Render transfer rate vs CKA similarity, per (surrogate, recipe)."""
    raise NotImplementedError


def plot_regression_ablation(
    regressors: dict[str, Any], ablation_results: dict[str, Any]
) -> Figure:
    """Render the regression fit and the CKA-guided-vs-random ablation comparison."""
    raise NotImplementedError
