"""Render manuscript artifacts from published aggregates without model access."""

from __future__ import annotations

from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter

from transfer_risk.lib.dbs import _diagonal_box_mask
from transfer_risk.lib.paper_reporting import (
    audit_transition_counts,
    exclude_surrogates,
    render_recipe_table,
    render_sensitivity_table,
    render_surrogate_table,
    render_transition_table,
)

_INK = "#263238"
_BLUE = "#476b87"
_GRAY = "#707070"
_DARK_CELL_THRESHOLD = 0.72


def build_pool_sensitivity(summary: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Recompute the configured surrogate exclusion from saved aggregate outcomes."""
    return exclude_surrogates(summary, params["excluded_surrogates"])


def build_paper_tables(
    summary: dict[str, Any],
    cells: pd.DataFrame,
    training: dict[str, Any],
    reduced: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[str, str, str, str]:
    """Prepare the manuscript's four catalog-owned Markdown table includes."""
    return (
        render_transition_table(summary),
        render_surrogate_table(
            summary, manifest["n_recipes"] * manifest["attack_attempts_per_cell"]
        ),
        render_recipe_table(cells),
        render_sensitivity_table(summary, training, reduced),
    )


def _box(
    axis: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    accent: bool = False,
) -> None:
    """Draw a labeled diagram box in normalized figure coordinates."""
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            facecolor="#edf2f5" if accent else "#fafafa",
            edgecolor=_BLUE if accent else "#b5b5b5",
            linewidth=1.1,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=10,
        color=_INK,
        linespacing=1.6,
    )


def _arrow(axis: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Connect diagram boxes with a directed arrow."""
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            color=_GRAY,
            linewidth=1.0,
            shrinkA=3,
            shrinkB=3,
        )
    )


def plot_audit_flow(summary: dict[str, Any], metrics: dict[str, Any]) -> Figure:
    """Show source eligibility and the target baselines defining the primary denominator."""
    transitions = audit_transition_counts(summary)
    cohort, attacks = summary["full_cohort"], metrics["attack_outcomes"]
    fig, axis = plt.subplots(figsize=(11.5, 6.4), layout="constrained")
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    _box(axis, 0.02, 0.79, 0.27, 0.13, f"Source evaluations\n{attacks['attempted']:,} records")
    _box(
        axis, 0.36, 0.79, 0.27, 0.13, f"Source initially positive\n{attacks['eligible']:,} eligible"
    )
    _box(
        axis,
        0.70,
        0.79,
        0.28,
        0.13,
        f"Source attack succeeds\n{attacks['successful']:,} records",
        accent=True,
    )
    _arrow(axis, (0.29, 0.855), (0.36, 0.855))
    _arrow(axis, (0.63, 0.855), (0.70, 0.855))
    axis.text(
        0.495,
        0.73,
        f"{attacks['skipped']:,} baseline misses skipped",
        ha="center",
        color=_GRAY,
        fontsize=9,
    )
    axis.text(
        0.84,
        0.73,
        f"{attacks['failed']:,} eligible attacks failed",
        ha="center",
        color=_GRAY,
        fontsize=9,
    )
    axis.text(
        0.5,
        0.61,
        "Target audit of successful source-attack records",
        ha="center",
        color=_INK,
        fontsize=12,
    )
    _box(
        axis,
        0.09,
        0.39,
        0.36,
        0.14,
        f"Original target prediction: injection\n{cohort['target_original_injection']:,} records",
        accent=True,
    )
    _box(
        axis,
        0.57,
        0.39,
        0.34,
        0.14,
        f"Original target prediction: benign\n{cohort['target_original_benign']:,} records",
    )
    _arrow(axis, (0.5, 0.58), (0.27, 0.54))
    _arrow(axis, (0.5, 0.58), (0.74, 0.54))
    _box(
        axis,
        0.02,
        0.13,
        0.23,
        0.14,
        f"Perturbed benign\n{transitions['injection_to_benign']:,} verified flips",
        accent=True,
    )
    _box(
        axis,
        0.29,
        0.13,
        0.23,
        0.14,
        f"Perturbed injection\n{transitions['injection_to_injection']:,} records",
    )
    _box(
        axis,
        0.57,
        0.13,
        0.19,
        0.14,
        f"Perturbed benign\n{transitions['benign_to_benign']:,} records",
    )
    _box(
        axis,
        0.80,
        0.13,
        0.18,
        0.14,
        f"Perturbed injection\n{transitions['benign_to_injection']:,} records",
    )
    for start, end in [
        ((0.27, 0.39), (0.135, 0.28)),
        ((0.27, 0.39), (0.405, 0.28)),
        ((0.74, 0.39), (0.665, 0.28)),
        ((0.74, 0.39), (0.89, 0.28)),
    ]:
        _arrow(axis, start, end)
    rates = (
        f"Primary rate: {cohort['true_target_flips']:,} / {cohort['target_original_injection']:,} "
        f"= {100 * cohort['true_target_flip_rate']:.1f}%     "
        f"Historical conditional rate: {cohort['target_perturbed_benign']:,} / "
        f"{cohort['source_successful']:,} "
        f"= {100 * cohort['conditional_target_benign_rate']:.1f}%"
    )
    axis.text(0.5, 0.04, rates, ha="center", color=_INK, fontsize=10)
    return fig


def plot_recipe_heatmap(cells: pd.DataFrame) -> Figure:
    """Plot each audited cell's rate and numerator/denominator without prompt text."""
    names = cells.groupby("surrogate")["mean_cka"].first().sort_values(ascending=False).index
    recipes = ["deepwordbug", "pwws", "textfooler", "bae", "bert-attack"]
    ordered = cells.set_index(["surrogate", "recipe"])
    rates = np.array(
        [
            [ordered.loc[(name, recipe), "true_target_flip_rate"] for recipe in recipes]
            for name in names
        ],
        dtype=float,
    )
    fig, axis = plt.subplots(figsize=(10.5, 7), layout="constrained")
    image = axis.imshow(np.ma.masked_invalid(rates), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set(
        xticks=range(len(recipes)),
        xticklabels=["DeepWordBug", "PWWS", "TextFooler", "BAE", "BERT-Attack"],
        yticks=range(len(names)),
        yticklabels=names,
    )
    axis.tick_params(axis="both", labelsize=10.5, length=0, pad=8)
    for i, name in enumerate(names):
        for j, recipe in enumerate(recipes):
            row = ordered.loc[(name, recipe)]
            rate = rates[i, j]
            label = (
                f"{100 * rate:.1f}%\n{int(row['true_target_flips'])} / "
                f"{int(row['target_original_injection'])}"
                if np.isfinite(rate)
                else "Undefined\n0 target injections"
            )
            axis.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=10.5,
                color="white" if rate >= _DARK_CELL_THRESHOLD else "black",
                linespacing=1.5,
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.75, pad=0.025)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    colorbar.set_label("Verified injection-to-benign rate", fontsize=9)
    axis.set_title("Audited outcomes by surrogate and attack recipe", fontsize=12, pad=15)
    return fig


def plot_dbs_geometry(similarity: dict[str, Any]) -> Figure:
    """Illustrate selector geometry with synthetic masks and no saved CKA values."""
    box = int(similarity["dbs"]["box"])
    shapes = [(12, 12), (12, 6), (12, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), layout="constrained", width_ratios=[12, 6, 2])
    for axis, (rows, columns) in zip(axes, shapes, strict=True):
        mask = _diagonal_box_mask(rows, columns, box)
        axis.imshow(mask, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        axis.set(
            xticks=range(columns),
            xticklabels=range(1, columns + 1),
            yticks=range(rows),
            yticklabels=range(1, rows + 1),
            title=f"12-by-{columns}\n{int(mask.sum())} / {mask.size}",
        )
        axis.set_xticks(np.arange(-0.5, columns), minor=True)
        axis.set_yticks(np.arange(-0.5, rows), minor=True)
        axis.grid(which="minor", color="white", linewidth=1)
        axis.tick_params(which="both", length=0, labelsize=9)
    axes[0].set_ylabel("Target representation")
    fig.supxlabel("Source representation", fontsize=11)
    fig.suptitle(f"DBS pair selection with box half-width {box}", fontsize=13)
    return fig


def plot_probe_stability(summary: dict[str, Any], training: pd.DataFrame) -> Figure:
    """Compare original and training-only CKA values for the same saved models."""
    rows = pd.DataFrame(summary["full_cohort"]["surrogate_macro"]["rows"])
    if set(rows["surrogate"]) != set(training["surrogate"]):
        msg = "original and training probes must contain the same surrogate identifiers"
        raise ValueError(msg)
    rows = rows.merge(
        training[["surrogate", "mean_cka"]],
        on="surrogate",
        suffixes=("", "_train"),
        validate="one_to_one",
    )
    rows = rows.sort_values("mean_cka", ascending=False).reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(9.5, 5.8), layout="constrained")
    y = np.arange(len(rows))
    for position, (_, row) in zip(y, rows.iterrows(), strict=True):
        axis.plot(
            [row["mean_cka"], row["mean_cka_train"]],
            [position, position],
            color="#a0a0a0",
            linewidth=1.5,
        )
    axis.scatter(
        rows["mean_cka"], y, marker="o", s=48, color=_BLUE, label="Original probe (2,000)", zorder=3
    )
    axis.scatter(
        rows["mean_cka_train"],
        y,
        marker="D",
        s=48,
        facecolors="white",
        edgecolors=_INK,
        label="Training-only probe (1,600)",
        zorder=4,
    )
    axis.set(
        yticks=y,
        yticklabels=rows["surrogate"],
        xlabel="Mean linear CKA",
        title="CKA ordering under the training-only probe",
    )
    axis.invert_yaxis()
    axis.tick_params(labelsize=9)
    axis.grid(axis="x", color="#dddddd", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", fontsize=9, frameon=False)
    return fig
