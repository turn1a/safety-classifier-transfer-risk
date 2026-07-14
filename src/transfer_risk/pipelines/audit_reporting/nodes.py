"""Nodes for publishing target-free target-audit artifacts.

The explicit audit-reporting pipeline consumes finalized, prediction-derived aggregates only.
It never accesses the frozen target, attack records, source prompt text, or surrogate models.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from transfer_risk.lib.audit_reporting import (
    export_public_target_audit_cells,
    export_public_target_audit_sources,
    prepare_true_flip_ablation_data,
    prepare_true_flip_scatter_data,
    publish_safe_audit_summary,
)

logger = logging.getLogger(__name__)
_GROUP_COLORS = {"M1": "#2a9d8f", "M2": "#999999"}
_GROUP_MARKERS = {"M1": "o", "M2": "s"}
_RECIPE_STYLES = {
    "bae": ("#0072B2", "o"),
    "bert-attack": ("#D55E00", "s"),
    "deepwordbug": ("#009E73", "^"),
    "pwws": ("#CC79A7", "D"),
    "textfooler": ("#E69F00", "P"),
}
_SCATTER_LABEL_OFFSETS = {
    "deberta-base-ft-seed": (-112, 9),
    "deberta-base-pi-v1": (8, -15),
    "llama-prompt-guard-22m": (-100, 12),
    "bilstm-attention": (8, 9),
}


def build_public_target_audit_cells(target_audit_cells: pd.DataFrame) -> pd.DataFrame:
    """Build the 50-cell aggregate-only public target-audit CSV.

    Args:
        target_audit_cells: Stable corrected audit cells from the catalog.

    Returns:
        Public corrected audit cells without prompt text.
    """
    cells = export_public_target_audit_cells(target_audit_cells)
    logger.info("Prepared %d public corrected target-audit cells", len(cells))
    return cells


def build_public_target_audit_sources(target_audit_raw_sources: pd.DataFrame) -> pd.DataFrame:
    """Build the aggregate-only public target-audit source CSV.

    Args:
        target_audit_raw_sources: Stable source-level audit aggregates from the catalog.

    Returns:
        Public source aggregates without prompt text.
    """
    sources = export_public_target_audit_sources(target_audit_raw_sources)
    logger.info("Prepared %d public target-audit source aggregate rows", len(sources))
    return sources


def build_public_target_audit_summary(target_audit_summary: dict[str, Any]) -> dict[str, Any]:
    """Publish the finalized corrected target-audit summary without rewriting its analyses.

    Args:
        target_audit_summary: Stable finalized audit summary from the catalog.

    Returns:
        Strict JSON-safe copy of the full finalized summary.
    """
    summary = publish_safe_audit_summary(target_audit_summary)
    logger.info(
        "Prepared public target-audit summary: %d source successes and %d true target flips",
        summary["full_cohort"]["source_successful"],
        summary["full_cohort"]["true_target_flips"],
    )
    return summary


def plot_true_flip_scatter(
    target_audit_cells: pd.DataFrame,
    target_audit_summary: dict[str, Any],
) -> Figure:
    """Render full-cohort macro and recipe true-target-flip rates against mean CKA.

    Args:
        target_audit_cells: Stable corrected audit cell aggregates.
        target_audit_summary: Stable finalized audit summary.

    Returns:
        Matplotlib figure for ``docs/figures/fig_true_flip_scatter.png``.
    """
    macro_rows, recipe_rows, association = prepare_true_flip_scatter_data(
        target_audit_cells,
        target_audit_summary,
    )
    fig, ax = plt.subplots(figsize=(9.2, 6.8), layout="constrained")
    for recipe, recipe_frame in recipe_rows.groupby("recipe", sort=True):
        recipe_name = str(recipe)
        if recipe_name not in _RECIPE_STYLES:
            msg = f"no audit-reporting style configured for recipe {recipe_name!r}"
            raise ValueError(msg)
        color, marker = _RECIPE_STYLES[recipe_name]
        ax.scatter(
            recipe_frame["mean_cka"],
            recipe_frame["true_target_flip_rate"],
            alpha=0.32,
            s=34,
            color=color,
            marker=marker,
            label=recipe_name,
        )
    ax.scatter(
        macro_rows["mean_cka"],
        macro_rows["true_target_flip_rate"],
        s=140,
        color="#1d3557",
        marker="o",
        edgecolors="white",
        linewidths=0.9,
        zorder=3,
        label="surrogate macro true flip rate",
    )
    for _, row in macro_rows.iterrows():
        surrogate = str(row["surrogate"])
        if surrogate not in _SCATTER_LABEL_OFFSETS:
            continue
        ax.annotate(
            surrogate,
            (row["mean_cka"], row["true_target_flip_rate"]),
            textcoords="offset points",
            xytext=_SCATTER_LABEL_OFFSETS[surrogate],
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.7},
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.75},
        )
    annotation = (
        f"Spearman rho = {association['rho']:.2f}; "
        f"exact-enumeration two-sided p = {association['two_sided_p']:.4f}; "
        f"n = {association['n']}"
    )
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax.set(
        xlabel="mean CKA (surrogate-level)",
        ylabel="true target flip rate",
        title="True target flip rate vs mean CKA",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=8,
        frameon=True,
    )
    fig.text(
        0.5,
        0.08,
        "Designed surrogate pool; exact p-value assumes an exchangeability null.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    return fig


def plot_true_flip_ablation(
    target_audit_summary: dict[str, Any],
    surrogate_selection: dict[str, Any],
) -> Figure:
    """Render the six M1/M2 macro true-target-flip rates and exact ablation context.

    Args:
        target_audit_summary: Stable finalized audit summary.
        surrogate_selection: Saved original-CKA M1/M2 membership.

    Returns:
        Matplotlib figure for ``docs/figures/fig_true_flip_ablation.png``.
    """
    ablation_rows, stats = prepare_true_flip_ablation_data(
        target_audit_summary,
        surrogate_selection,
    )
    fig, ax = plt.subplots(figsize=(10.2, 5.8), layout="constrained")
    group_positions = {"M1": 0.0, "M2": 1.0}
    for group in ("M1", "M2"):
        group_rows = ablation_rows.loc[ablation_rows["similarity_group"] == group]
        offsets = np.linspace(-0.08, 0.08, len(group_rows))
        ax.scatter(
            group_positions[group] + offsets,
            group_rows["true_target_flip_rate"],
            s=96,
            color=_GROUP_COLORS[group],
            marker=_GROUP_MARKERS[group],
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
            label=group,
        )
        for offset, (_, row) in zip(offsets, group_rows.iterrows(), strict=True):
            surrogate = str(row["surrogate"])
            text_offset = (7, 0) if group == "M1" else (-7, 0)
            if surrogate == "llama-prompt-guard-22m":
                text_offset = (-7, 10)
            ax.annotate(
                surrogate,
                (group_positions[group] + offset, row["true_target_flip_rate"]),
                textcoords="offset points",
                xytext=text_offset,
                ha="left" if group == "M1" else "right",
                fontsize=8,
            )
    ax.scatter(
        [group_positions["M1"], group_positions["M2"]],
        [stats["m1_mean"], stats["m2_mean"]],
        s=175,
        color=["#264653", "#6d6875"],
        marker="*",
        zorder=4,
        label="group mean",
    )
    annotation = (
        f"+{stats['mean_diff_pp']:.1f} pp\n"
        f"exact one-sided p = {_format_probability(stats['mean_p_value'])}\n"
        "1/20 floor; strict p < .05 rule not met"
    )
    ax.text(
        0.98,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax.set_xticks([group_positions["M1"], group_positions["M2"]], labels=["M1", "M2"])
    ax.set(
        title="Selection ablation: true target flip rate",
        ylabel="macro true target flip rate",
        xlabel="similarity group",
        xlim=(-0.5, 1.5),
    )
    ax.legend(loc="lower left", ncol=3, fontsize=8)
    return fig


def _format_probability(value: float | int) -> str:
    """Format a two-decimal probability without a leading zero below one."""
    return f"{float(value):.2f}".removeprefix("0")
