"""Nodes for the reporting pipeline (SPEC.md §10).

Each node returns a Matplotlib ``Figure`` that the catalog persists via
``matplotlib.MatplotlibDataset``. The Agg backend is selected so the figures render
headless (CI, ``kedro run`` without a display).
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def plot_cka_heatmap(similarity_table: pd.DataFrame) -> Figure:
    """Render a heatmap of the per-surrogate similarity scalars (mean CKA, DBS)."""
    fig, ax = plt.subplots(figsize=(6, 0.6 * len(similarity_table) + 2))
    data = similarity_table.set_index("surrogate")[["mean_cka", "dbs"]]
    sns.heatmap(data, annot=True, fmt=".3f", cmap="viridis", vmin=0.0, vmax=1.0, ax=ax)
    ax.set(title="Target-vs-surrogate similarity", xlabel="", ylabel="surrogate")
    fig.tight_layout()
    return fig


def plot_transfer_scatter(master_results_table: pd.DataFrame) -> Figure:
    """Scatter of transfer rate vs mean CKA, coloured by attack recipe."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=master_results_table, x="mean_cka", y="transfer_rate", hue="recipe", s=90, ax=ax
    )
    ax.set(xlabel="mean CKA", ylabel="transfer rate", title="Transfer rate vs CKA similarity")
    fig.tight_layout()
    return fig


def plot_regression_ablation(
    regressors: dict[str, Any], ablation_results: dict[str, Any]
) -> Figure:
    """Feature importances and the CKA-guided-vs-random max-transfer comparison."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))
    ax_left.bar(regressors["feature_names"], regressors["random_forest_importances"])
    ax_left.set(title="Random-forest feature importance", ylabel="importance")
    guided = ablation_results.get("guided_max_transfer", 0.0)
    random_mean = ablation_results.get("random_max_mean", 0.0)
    ax_right.bar(["CKA-guided", "random"], [guided, random_mean], color=["#2a9d8f", "#999999"])
    effect = ablation_results.get("effect_size_pp", 0.0)
    ax_right.set(title=f"Selection ablation ({effect:+.1f}pp)", ylabel="max transfer rate")
    fig.tight_layout()
    return fig
