"""Tests for target-free manuscript reporting and complete model comparisons."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from transfer_risk.pipeline_registry import register_pipelines
from transfer_risk.pipelines.paper_reporting.nodes import (
    build_paper_tables,
    build_review_reporting,
    plot_audit_flow,
    plot_objective_sensitivity,
    plot_probe_stability,
)

_ROOT = Path(__file__).resolve().parents[2]


def _artifact(name: str) -> dict:
    """Read a public aggregate fixture without experiment data."""
    return json.loads((_ROOT / "docs/artifacts" / name).read_text())


def test_paper_reporting_has_only_public_aggregate_inputs() -> None:
    """Manuscript regeneration cannot acquire models or execute the experiment."""
    pipelines = register_pipelines()
    reporting = pipelines["paper_reporting"]
    assert all(name.startswith(("pub_", "params:")) for name in reporting.inputs())
    names = {node.name for node in reporting.nodes}
    for composite in ("__default__", "stage", "downstream", "audit"):
        assert not names & {node.name for node in pipelines[composite].nodes}


def test_tables_reconcile_saved_counts_and_sensitivity() -> None:
    """The table boundary retains distinct transition, macro, and pooled outcomes."""
    tables = build_paper_tables(
        _artifact("target_audit_summary.json"),
        pd.read_csv(_ROOT / "docs/artifacts/target_audit_cells.csv"),
        _artifact("training_probe_similarity_sensitivity.json"),
        _artifact("surrogate_excluded_sensitivity.json"),
        _artifact("results_manifest.json"),
    )
    transitions, surrogates, recipes, sensitivities = tables
    assert "| Injection | 1,054 | 2,955 | 4,009 |" in transitions
    assert "| Total | 1,398 | 2,964 | 4,362 |" in transitions
    assert "67 / 124 | 53.5 | 54.0 | 7.0" in surrogates
    assert "bae" in recipes
    assert "998 / 3,410 | 29.3 | 0.800 | 0.014" in sensitivities


def test_probe_plot_rejects_missing_surrogates() -> None:
    """A stale training table cannot silently omit a model from the comparison."""
    training = pd.read_csv(_ROOT / "docs/artifacts/training_probe_similarity.csv").iloc[:-1]
    try:
        with pytest.raises(ValueError, match="same surrogate"):
            plot_probe_stability(_artifact("target_audit_summary.json"), training)
    finally:
        plt.close("all")


@pytest.mark.parametrize("case", ["undefined", "zero", "empty", "probability"])
def test_objective_plot_handles_undefined_and_empty_outcomes(case: str) -> None:
    """Nullable outcomes and zero-size ranges produce readable figures without warnings."""
    objectives = _artifact("pool_objective_sensitivity.json")
    comparison = _artifact("checkpoint_comparison.json")
    pool = objectives["pools"]["surrogate_excluded"]
    for row in objectives["surrogate_rows"]:
        row["macro_conditional_verified_flip_rate"] = None if case == "undefined" else 0.0
        row["source_success_gated_verified_flip_yield"] = 0.0
    for row in comparison["reference_pool"]["surrogate_rows"]:
        row["all_candidate_yield"] = 0.0
    for stats in pool["associations"].values():
        stats.update(rho=None, two_sided_p=None, reason="constant or absent outcomes")
        if case == "probability":
            stats["rho"] = 0.5
    comparison["reference_pool"]["association"] = None
    if case == "empty":
        pool["surrogates"] = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        figure = plot_objective_sensitivity(objectives, comparison)
        figure.canvas.draw()
    try:
        labels = " ".join(text.get_text() for axis in figure.axes for text in axis.texts)
        assert "Undefined" in labels
        assert "constant or absent outcomes" in labels
        if case == "probability":
            assert "0.500" in labels
            assert "p = Undefined" in labels
        assert all(axis.get_ylim()[1] > 0 for axis in figure.axes)
    finally:
        plt.close(figure)


def test_audit_flow_explains_undefined_probabilities() -> None:
    """An empty cohort renders denominator reasons in both displayed rates."""
    summary = _artifact("target_audit_summary.json")
    for name in (
        "source_successful",
        "target_original_benign",
        "target_original_injection",
        "target_perturbed_benign",
        "true_target_flips",
    ):
        summary["full_cohort"][name] = 0
    summary["full_cohort"].update(true_target_flip_rate=None, conditional_target_benign_rate=None)
    figure = plot_audit_flow(summary, _artifact("run_metrics.json"))
    try:
        figure.canvas.draw()
        labels = " ".join(text.get_text() for text in figure.axes[0].texts)
        assert "Undefined (no original target injections)" in labels
        assert "Undefined (no successful source attacks)" in labels
    finally:
        plt.close(figure)


def test_review_reporting_preserves_pending_and_complete_unresolved_states() -> None:
    """Pending human outcomes never render as measured zero validity."""
    preparation = _artifact("semantic_flip_review_preparation.json")
    table, figure = build_review_reporting(preparation, None)
    try:
        labels = " ".join(text.get_text() for axis in figure.axes for text in axis.texts)
        assert "Pending" in table and "Pending" in labels
        assert "0.0%" not in table
        assert "300" in labels and "1,403" in labels and "291" in labels
        assert "1,027" in labels and "1,318" in labels
        assert "0.0 to 16.3" in table
        assert "0.0 to 11.6" in table
    finally:
        plt.close(figure)
    completed = {
        **preparation,
        "review_status": "complete",
        "validity_status": "unresolved",
        "reviewed_pairs": 1318,
        "two_reviewer_complete_pairs": 1318,
    }
    table, figure = build_review_reporting(preparation, completed)
    try:
        assert "Complete" in table and "Unresolved" in table
        assert "Pending" not in table
    finally:
        plt.close(figure)


def test_review_reporting_refuses_stale_final_summary() -> None:
    """A finalized artifact cannot override a different preparation population."""
    preparation = _artifact("semantic_flip_review_preparation.json")
    for field, value in (
        ("attempted_trials", 8596),
        ("stage_one_sample_trials", 299),
        ("review_preparation_id", "stale"),
    ):
        with pytest.raises(ValueError, match="preparation"):
            build_review_reporting(preparation, {**preparation, field: value})
