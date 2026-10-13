"""Tests for target-free manuscript reporting and complete model comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from transfer_risk.pipeline_registry import register_pipelines
from transfer_risk.pipelines.paper_reporting.nodes import build_paper_tables, plot_probe_stability

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
