"""Catalog-only preparation, fixed saved counts and separate review command wiring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from kedro.config import OmegaConfigLoader

from transfer_risk.lib.semantic_flip_audit import build_flip_cohort, maximum_interval_margin
from transfer_risk.pipeline_registry import register_pipelines
from transfer_risk.pipelines.semantic_flip_audit.nodes import prepare_flip_cohort
from transfer_risk.scripts.semantic_flip_review_server import default_port

ROOT = Path(__file__).parents[2]


def test_new_cohort_is_opt_in_and_public_outputs_are_wrapped() -> None:

    pipelines = register_pipelines()
    assert {"semantic_flip_audit", "semantic_flip_review", "semantic_flip_review_tools"}.issubset(
        pipelines
    )
    assert not set(pipelines["semantic_flip_audit"].nodes) & set(pipelines["__default__"].nodes)
    assert not any(
        "model" in name or "attack" in name for name in pipelines["semantic_flip_audit"].outputs()
    )
    config = OmegaConfigLoader(conf_source=str(ROOT / "conf"), base_env="base", env="local")[
        "catalog"
    ]
    for key in ["pub_semantic_flip_preparation", "pub_semantic_flip_review"]:
        assert config[key]["type"] == "kedro_mlflow.io.artifacts.MlflowArtifactDataset"
    assert "semantic_flip_audit" in config["semantic_flip_cohort"]["filepath"]


def test_actual_saved_population_and_historical_artifacts_are_preserved() -> None:

    pairs = pd.read_csv(
        ROOT / "data/08_reporting/private/semantic_audit/pairs.csv", keep_default_na=False
    )
    linkage = pd.read_csv(
        ROOT / "data/08_reporting/private/semantic_audit/sample_linkage.csv", keep_default_na=False
    )
    predictions = pd.read_csv(
        ROOT / "data/08_reporting/private/retrospective_audit/protectai-v2_predictions.csv",
        keep_default_na=False,
    )
    cohort = build_flip_cohort(pairs, linkage, predictions, seed=20260616, sample_size=300)
    assert cohort["attempted_trials"] == 8595
    assert len(cohort["trials"]) == 1403
    assert cohort["review_dataset"]["pair_count"] == 1318
    assert len(cohort["sample_occurrence_ids"]) == 300
    assert sum(row["source_success"] for row in cohort["trials"]) == 998
    assert maximum_interval_margin(1403, 300, 8595) <= 0.0085
    baseline = ROOT / ".superpowers/sdd/2026-10-15-review-corrections/baseline-artifact-hashes.json"
    if baseline.exists():
        for name, digest in json.loads(baseline.read_text()).items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def test_new_server_uses_separate_ports_and_refuses_role_or_cohort_mismatch() -> None:

    assert default_port("reviewer_1") == 8767
    assert default_port("reviewer_2") == 8768
    with pytest.raises(ValueError):
        default_port("outside")


def test_current_prediction_reduction_accepts_csv_rounding_and_refuses_changed_counts() -> None:
    pairs = pd.read_csv(
        ROOT / "data/08_reporting/private/semantic_audit/pairs.csv", keep_default_na=False
    )
    linkage = pd.read_csv(
        ROOT / "data/08_reporting/private/semantic_audit/sample_linkage.csv", keep_default_na=False
    )
    predictions = pd.read_csv(
        ROOT / "data/08_reporting/private/retrospective_audit/protectai-v2_predictions.csv",
        keep_default_na=False,
    )
    reference = pd.read_csv(ROOT / "docs/artifacts/retrospective_protectai-v2_cells.csv")
    historical = pd.read_csv(ROOT / "docs/artifacts/target_audit_cells.csv")
    manifest = json.loads((ROOT / "docs/artifacts/results_manifest.json").read_text())
    checked = {"inference_context_check": {"identity_and_settings_agree": True}}
    params = {
        "sample_trials": 300,
        "expected_population": {
            "attempted_trials": 8595,
            "changed_trials": 1403,
            "unique_pairs": 1318,
        },
    }
    result = prepare_flip_cohort(
        pairs, linkage, predictions, reference, historical, manifest, checked, params, 20260616
    )
    assert result["attempted_trials"] == 8595
    reference.loc[0, "verified_flips_all_candidates"] += 1
    with pytest.raises(ValueError, match="reference cells"):
        prepare_flip_cohort(
            pairs, linkage, predictions, reference, historical, manifest, checked, params, 20260616
        )
