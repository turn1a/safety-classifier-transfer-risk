"""Nodes for the reporting pipeline (SPEC.md §10).

Each node returns artifacts persisted through the catalog: Matplotlib figures under
``docs/figures/`` and the public CSV/JSON bundle under ``docs/artifacts/``. The Agg backend
is selected so figures render headless (CI, ``kedro run`` without a display).
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
from matplotlib.patches import Patch

from transfer_risk.lib.association import spearman_association
from transfer_risk.lib.public_bundle import (
    apply_corrected_dbs,
    build_model_validation_summary,
    build_public_run_metrics,
    build_redacted_qualitative_audit,
    build_results_manifest,
    build_surrogate_summary,
    export_public_master_table,
    publish_dataset_audit,
    summarize_attack_outcomes,
)
from transfer_risk.pipelines._dynamic import surrogate_specs

logger = logging.getLogger(__name__)
_GROUP_COLORS = {"M1": "#2a9d8f", "M2": "#999999", "middle": "#457b9d"}
_GROUP_MARKERS = {"M1": "o", "M2": "s", "middle": "D"}
_GROUP_HATCHES = {"M1": "///", "M2": "xx", "middle": ""}
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


def recompute_master_dbs(
    master_results_table: pd.DataFrame,
    cka_matrices: dict[str, Any],
    params_similarity: dict[str, Any],
) -> pd.DataFrame:
    """Recompute DBS from saved CKA matrices without mutating upstream artifacts.

    Args:
        master_results_table: Saved master results table.
        cka_matrices: Saved CKA matrices keyed by surrogate.
        params_similarity: Similarity-stage parameters (``dbs.box`` half-width).

    Returns:
        Copy of the master table with corrected ``dbs`` and ``n_target_benign``.
    """
    box = int(params_similarity["dbs"]["box"])
    corrected = apply_corrected_dbs(master_results_table, cka_matrices, box=box)
    logger.info("Recomputed corrected DBS for %d master rows (box=%d)", len(corrected), box)
    return corrected


def build_public_master_results_table(master_results_corrected: pd.DataFrame) -> pd.DataFrame:
    """Export the 50-row public master results table.

    Args:
        master_results_corrected: Corrected master table from :func:`recompute_master_dbs`.

    Returns:
        Public master table for ``docs/artifacts/master_results_table.csv``.
    """
    return export_public_master_table(master_results_corrected)


def build_surrogate_summary_table(
    master_results_corrected: pd.DataFrame,
    surrogate_selection: dict[str, Any],
) -> pd.DataFrame:
    """Build the 10-row surrogate summary table.

    Args:
        master_results_corrected: Corrected master table.
        surrogate_selection: Saved M1/M2 selection bands.

    Returns:
        Surrogate summary for ``docs/artifacts/surrogate_summary.csv``.
    """
    return build_surrogate_summary(master_results_corrected, surrogate_selection)


def publish_public_dataset_audit(dataset_audit: dict[str, Any]) -> dict[str, Any]:
    """Publish the safe dataset audit aggregate.

    Args:
        dataset_audit: Saved dataset audit JSON.

    Returns:
        Public dataset audit for ``docs/artifacts/dataset_audit.json``.
    """
    return publish_dataset_audit(dataset_audit)


def assemble_surrogate_metadata(**meta_fragments: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Merge catalog-wired surrogate metadata fragments by surrogate name.

    Args:
        **meta_fragments: One metadata dict per surrogate; Kedro wires each
            ``surrogate_meta__{name}`` catalog entry to a ``meta_{name}`` argument.

    Returns:
        Mapping ``surrogate -> metadata fragment``.
    """
    name_by_param = {
        f"meta_{spec['name'].replace('-', '_')}": spec["name"] for spec in surrogate_specs()
    }
    return {name_by_param[key]: value for key, value in meta_fragments.items()}


def build_model_validation_summary_node(
    surrogate_registry: dict[str, Any],
    surrogate_metadata_bundle: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the public model validation summary.

    Args:
        surrogate_registry: Saved surrogate registry.
        surrogate_metadata_bundle: Merged metadata fragments from the catalog.

    Returns:
        Model validation summary for ``docs/artifacts/model_validation_summary.json``.
    """
    return build_model_validation_summary(surrogate_registry, surrogate_metadata_bundle)


def build_public_run_metrics_bundle(
    master_results_corrected: pd.DataFrame,
    adversarial_examples: dict[str, list[dict[str, Any]]],
    ablation_results: dict[str, Any],
    thresholds: dict[str, float],
    params_attacks: dict[str, Any],
) -> dict[str, Any]:
    """Build corrected public run metrics from saved artifacts.

    Args:
        master_results_corrected: Corrected master table.
        adversarial_examples: Catalog-wired attack records, used only for aggregate
            result-type counts.
        ablation_results: Saved ablation result dict.
        thresholds: Calibrated thresholds.
        params_attacks: Attack-stage parameters (attempts per cell).

    Returns:
        Public run metrics for ``docs/artifacts/run_metrics.json``.
    """
    metrics = build_public_run_metrics(
        master_results_corrected,
        ablation_results,
        thresholds,
    )
    metrics["counts"]["attack_attempts_per_cell"] = int(params_attacks["eval_set_size"])
    metrics["counts"]["query_budget"] = int(params_attacks["query_budget"])
    metrics["attack_outcomes"] = summarize_attack_outcomes(adversarial_examples)
    return metrics


def build_results_manifest_node(
    params_reporting: dict[str, Any],
    params_seed: int,
    params_models: dict[str, Any],
    params_attacks: dict[str, Any],
    params_similarity: dict[str, Any],
    params_transfer: dict[str, Any],
) -> dict[str, Any]:
    """Build the safe public results manifest.

    Args:
        params_reporting: Reporting-stage manifest parameters.
        params_seed: Root reproducibility seed.
        params_models: Models-stage parameters (target id, surrogate count).
        params_attacks: Attack-stage parameters.
        params_similarity: Similarity-stage parameters.
        params_transfer: Transfer-stage parameters.

    Returns:
        Results manifest for ``docs/artifacts/results_manifest.json``.
    """
    manifest_cfg = params_reporting["manifest"]
    return build_results_manifest(
        {
            "schema_version": str(manifest_cfg["schema_version"]),
            "experiment_commit": str(manifest_cfg["experiment_commit"]),
            "root_seed": int(params_seed),
            "uv_lock_sha256": str(manifest_cfg["uv_lock_sha256"]),
            "target": str(params_models["target"]),
            "probe_window": {
                "n_probe": int(params_similarity["n_probe"]),
                "max_seq_len": int(params_similarity["max_seq_len"]),
            },
            "attack_window": {
                "eval_set_size": int(params_attacks["eval_set_size"]),
                "max_seq_len": int(params_attacks["max_seq_len"]),
                "query_budget": int(params_attacks["query_budget"]),
            },
            "transfer_window": {"max_seq_len": int(params_transfer["max_seq_len"])},
            "attack_attempts_per_cell": int(params_attacks["eval_set_size"]),
            "query_budget": int(params_attacks["query_budget"]),
            "n_surrogates": len(params_models["surrogates"]),
            "n_recipes": len(params_attacks["recipes"]),
            "hardware": dict(manifest_cfg["hardware"]),
            "completion_note": str(manifest_cfg["completion_note"]),
        }
    )


def build_qualitative_examples() -> list[dict[str, Any]]:
    """Build the redacted qualitative audit for public release.

    Returns:
        Fixed safe audit records for ``docs/artifacts/qualitative_examples.json``.
    """
    return build_redacted_qualitative_audit()


def _surrogate_level_table(master: pd.DataFrame) -> pd.DataFrame:
    """Aggregate corrected master rows to one row per surrogate.

    Args:
        master: Corrected master table.

    Returns:
        Surrogate-level summary with macro mean/min/max transfer.
    """
    return (
        master.groupby("surrogate", as_index=False)
        .agg(
            mean_cka=("mean_cka", "first"),
            dbs=("dbs", "first"),
            macro_mean_transfer=("transfer_rate", "mean"),
            macro_min_transfer=("transfer_rate", "min"),
            macro_max_transfer=("transfer_rate", "max"),
        )
        .sort_values("mean_cka")
        .reset_index(drop=True)
    )


def _similarity_groups(master: pd.DataFrame, selection: dict[str, Any]) -> pd.Series:
    """Return a similarity-group label per surrogate row.

    Args:
        master: Corrected master table or surrogate-level table with ``surrogate``.
        selection: Saved M1/M2 selection bands.

    Returns:
        Series of ``M1`` / ``M2`` / ``middle`` labels aligned with ``master``.
    """
    m1 = set(selection.get("M1", []))
    m2 = set(selection.get("M2", []))
    return master["surrogate"].map(
        lambda name: "M1" if name in m1 else ("M2" if name in m2 else "middle")
    )


def plot_cka_ranked(
    master_results_corrected: pd.DataFrame,
    surrogate_selection: dict[str, Any],
    thresholds: dict[str, float],
) -> Figure:
    """Render a ranked two-panel CKA/DBS view with M1/M2/middle encoding.

    Args:
        master_results_corrected: Corrected master table.
        surrogate_selection: Saved M1/M2 selection bands.
        thresholds: Calibrated ``{"r1", "r2"}`` thresholds.

    Returns:
        Matplotlib figure for ``docs/figures/fig_cka_ranked.png``.
    """
    summary = _surrogate_level_table(master_results_corrected)
    summary["similarity_group"] = _similarity_groups(summary, surrogate_selection)
    summary = summary.sort_values("mean_cka").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True, layout="constrained")
    y_positions = np.arange(len(summary))
    for ax, metric, title in zip(
        axes,
        ("mean_cka", "dbs"),
        ("mean CKA: r2 dotted / r1 dashed", "corrected DBS"),
        strict=True,
    ):
        colors = [_GROUP_COLORS[group] for group in summary["similarity_group"]]
        bars = ax.barh(y_positions, summary[metric], color=colors, alpha=0.85)
        for bar, group in zip(bars, summary["similarity_group"], strict=True):
            bar.set_hatch(_GROUP_HATCHES[group])
        ax.set(title=title, xlabel=title, yticks=y_positions, yticklabels=summary["surrogate"])
        ax.set_xlim(0.0, 1.0)
    axes[0].set_xlabel("mean CKA")
    axes[0].axvline(thresholds["r1"], color="#e76f51", linestyle="--", linewidth=1.2)
    axes[0].axvline(thresholds["r2"], color="#6d597a", linestyle=":", linewidth=1.5)
    legend_handles = [
        Patch(
            facecolor=_GROUP_COLORS[group],
            hatch=_GROUP_HATCHES[group],
            label=group,
            alpha=0.85,
        )
        for group in ("M1", "middle", "M2")
    ]
    axes[0].legend(handles=legend_handles, loc="lower right", title="similarity group")
    fig.suptitle("Target-vs-surrogate similarity (ranked by mean CKA)")
    return fig


def plot_transfer_scatter(master_results_corrected: pd.DataFrame) -> Figure:
    """Plot historical conditional target-benign rates against mean CKA.

    Args:
        master_results_corrected: Corrected master table.

    Returns:
        Matplotlib figure for ``docs/figures/fig_transfer_scatter.png``.
    """
    summary = _surrogate_level_table(master_results_corrected)
    stats = spearman_association(summary["mean_cka"], summary["macro_mean_transfer"])
    fig, ax = plt.subplots(figsize=(9.2, 6.8), layout="constrained")
    for recipe, recipe_frame in master_results_corrected.groupby("recipe"):
        color, marker = _RECIPE_STYLES[str(recipe)]
        ax.scatter(
            recipe_frame["mean_cka"],
            recipe_frame["transfer_rate"],
            alpha=0.32,
            s=32,
            color=color,
            marker=marker,
            label=recipe,
        )
    ax.scatter(
        summary["mean_cka"],
        summary["macro_mean_transfer"],
        s=135,
        color="#1d3557",
        marker="o",
        edgecolors="white",
        linewidths=0.9,
        zorder=3,
        label="surrogate macro historical conditional rate",
    )
    for _, row in summary.iterrows():
        surrogate = str(row["surrogate"])
        if surrogate not in _SCATTER_LABEL_OFFSETS:
            continue
        ax.annotate(
            surrogate,
            (row["mean_cka"], row["macro_mean_transfer"]),
            textcoords="offset points",
            xytext=_SCATTER_LABEL_OFFSETS[surrogate],
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.7},
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.75},
        )
    ax.set(
        xlabel="mean CKA (surrogate-level)",
        ylabel="Historical conditional target-benign rate",
        title="Historical conditional target-benign rate vs mean CKA",
    )
    ax.text(
        0.03,
        0.97,
        (
            f"Spearman rho = {stats['rho']:.2f}; "
            f"exact two-sided p = {stats['two_sided_p']:.3f}; "
            f"n = {stats['n']}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
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
        (
            "Historical conditional target-benign rate: "
            "P[target benign on perturbed prompt | source success].\n"
            "Designed surrogate pool; exact p-value assumes an exchangeability null."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    return fig
