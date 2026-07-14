"""Focused tests for risk-node association aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from kedro.config import OmegaConfigLoader

from transfer_risk.pipelines.risk import nodes as risk_nodes
from transfer_risk.pipelines.risk.pipeline import create_pipeline

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _risk_params() -> dict[str, Any]:
    """Return a minimal risk parameter block for node-unit tests."""
    return {
        "decision_tree": {"max_depth": 2},
        "random_forest": {"n_estimators": 5, "max_depth": 2},
        "ablation": {"n_permutations": 10, "alpha": 0.05},
    }


def test_fit_regressors_primary_association_is_surrogate_level() -> None:
    master = pd.DataFrame(
        [
            {"surrogate": "a", "recipe": "r1", "mean_cka": 0.2, "dbs": 0.2, "transfer_rate": 0.1},
            {"surrogate": "a", "recipe": "r2", "mean_cka": 0.2, "dbs": 0.2, "transfer_rate": 0.5},
            {"surrogate": "b", "recipe": "r1", "mean_cka": 0.4, "dbs": 0.4, "transfer_rate": 0.6},
            {"surrogate": "c", "recipe": "r1", "mean_cka": 0.8, "dbs": 0.8, "transfer_rate": 0.9},
        ]
    )
    result = risk_nodes.fit_regressors(master, _risk_params(), seed=7)
    assert result["surrogate_association"]["mean_cka"]["n"] == 3
    assert result["surrogate_association"]["dbs"]["n"] == 3
    assert result["recipe_association"]["mean_cka"]["n"] == 4
    assert result["recipe_association"]["dbs"]["n"] == 4


def test_recompute_risk_master_dbs_uses_saved_cka_without_mutating_master() -> None:
    """Recompute the risk input DBS from saved CKA matrices before statistics."""
    master = pd.DataFrame(
        [
            {
                "surrogate": "alpha",
                "recipe": "r1",
                "n_successful": 10,
                "transfer_rate": 0.2,
                "mean_cka": 0.8,
                "dbs": 0.1,
            },
            {
                "surrogate": "beta",
                "recipe": "r1",
                "n_successful": 20,
                "transfer_rate": 0.25,
                "mean_cka": 0.3,
                "dbs": 0.9,
            },
        ]
    )
    matrices = {
        "alpha": np.array([[1.0, 0.2], [0.3, 0.8]]),
        "beta": np.array([[0.9, 0.1], [0.4, 0.7]]),
    }

    corrected = risk_nodes.recompute_risk_master_dbs(
        master,
        matrices,
        {"dbs": {"box": 0}},
    )

    assert corrected["dbs"].tolist() == pytest.approx([0.9, 0.8])
    assert master["dbs"].tolist() == pytest.approx([0.1, 0.9])
    assert corrected["n_target_benign"].tolist() == [2, 5]


def test_risk_pipeline_uses_corrected_master_for_all_statistics() -> None:
    """Feed regressors, ablation, and metrics from one saved corrected risk table."""
    pipeline = create_pipeline()
    correction = next(item for item in pipeline.nodes if item.name == "recompute_risk_master_dbs")
    consumers = {
        item.name: item
        for item in pipeline.nodes
        if item.name in {"fit_regressors", "run_ablation", "track_run_metrics"}
    }

    assert correction.inputs == ["master_results_table", "cka_matrices", "params:similarity"]
    assert correction.outputs == ["risk_master_results_corrected"]
    assert set(consumers) == {"fit_regressors", "run_ablation", "track_run_metrics"}
    assert all("risk_master_results_corrected" in item.inputs for item in consumers.values())
    assert all("master_results_table" not in item.inputs for item in consumers.values())


def test_corrected_risk_master_is_catalog_owned() -> None:
    """Persist the corrected risk table without replacing the original master table."""
    loader = OmegaConfigLoader(
        conf_source=str(_PROJECT_ROOT / "conf"),
        base_env="base",
        default_run_env="local",
    )
    catalog = loader["catalog"]
    entry = catalog["risk_master_results_corrected"]

    assert entry["type"] == "kedro_mlflow.io.artifacts.MlflowArtifactDataset"
    assert entry["dataset"]["type"] == "pandas.ParquetDataset"
    assert (
        entry["dataset"]["filepath"] == "data/07_model_output/risk_master_results_corrected.parquet"
    )
    assert "master_results_table" in catalog


def test_track_run_metrics_keeps_primary_and_recipe_association_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(risk_nodes, "_log_mlflow_metrics", lambda _metrics: None)
    regressors = {
        "feature_names": ["mean_cka", "dbs"],
        "random_forest_importances": [0.3, 0.7],
        "surrogate_association": {
            "mean_cka": {"rho": 0.75, "two_sided_p": 0.01, "n": 10, "exact": True},
            "dbs": {"rho": 0.50, "two_sided_p": 0.20, "n": 10, "exact": True},
        },
        "recipe_association": {
            "mean_cka": {"rho": 0.40, "two_sided_p": 0.11, "n": 20, "exact": False},
            "dbs": {"rho": 0.30, "two_sided_p": 0.25, "n": 20, "exact": False},
        },
    }
    master = pd.DataFrame(
        [{"surrogate": "a", "recipe": "r1", "mean_cka": 0.2, "dbs": 0.2, "transfer_rate": 0.1}]
    )
    metrics = risk_nodes.track_run_metrics(
        master=master,
        ablation={"effect_size_pp": 12.0, "empirical_p_value": 0.2},
        regressors=regressors,
        thresholds={"r1": 0.9, "r2": 0.1},
    )
    assert metrics["surrogate_spearman_mean_cka_rho"] == pytest.approx(0.75)
    assert metrics["surrogate_spearman_mean_cka_p"] == pytest.approx(0.01)
    assert metrics["recipe_spearman_mean_cka_rho"] == pytest.approx(0.40)
    assert metrics["recipe_spearman_mean_cka_p"] == pytest.approx(0.11)
    # Backward-compatible headline key now points at the primary surrogate-level result.
    assert metrics["spearman_mean_cka_rho"] == pytest.approx(0.75)
    assert metrics["spearman_mean_cka_p"] == pytest.approx(0.01)
