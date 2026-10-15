"""Catalog-only preparation, fixed saved counts and separate review command wiring."""

from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from kedro.config import OmegaConfigLoader

from transfer_risk.lib.retrospective_audit import build_retrospective_cells
from transfer_risk.lib.semantic_audit import stable_pair_id
from transfer_risk.lib.semantic_flip_audit import build_flip_cohort, maximum_interval_margin
from transfer_risk.pipeline_registry import register_pipelines
from transfer_risk.pipelines.semantic_flip_audit.nodes import prepare_flip_cohort
from transfer_risk.scripts.semantic_flip_review_server import default_port

ROOT = Path(__file__).parents[2]


def _load_private_saved_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run optional local evidence checks only when every private input exists."""
    paths = [
        ROOT / "data/08_reporting/private/semantic_audit/pairs.csv",
        ROOT / "data/08_reporting/private/semantic_audit/sample_linkage.csv",
        ROOT / "data/08_reporting/private/retrospective_audit/protectai-v2_predictions.csv",
    ]
    if not all(path.is_file() for path in paths):
        pytest.skip("optional local saved-data check requires ignored private experiment inputs")
    pairs, linkage, predictions = [pd.read_csv(path, keep_default_na=False) for path in paths]
    return pairs, linkage, predictions


def _synthetic_saved_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build tracked synthetic counts with a one-in-seven derived rate."""
    pairs = pd.DataFrame(
        [
            {
                "pair_id": stable_pair_id(f"synthetic original {index}", f"synthetic edit {index}"),
                "original_id": f"synthetic-{index}",
                "original": f"synthetic original {index}",
                "perturbed": f"synthetic edit {index}",
            }
            for index in range(3)
        ]
    )
    linkage = pd.DataFrame(
        [
            {
                "pair_id": pairs.iloc[index].pair_id,
                "original_id": pairs.iloc[index].original_id,
                "source": "synthetic",
                "surrogate": model,
                "recipe": "synthetic-recipe",
                "attempt_count": 7,
                "source_successful_count": 1,
                "source_failed_count": 6,
                "source_skipped_count": 0,
                "source_other_count": 0,
            }
            for index, model in enumerate(["synthetic-a", "synthetic-b", "bilstm-attention"])
        ]
    )
    predictions = pairs[["pair_id", "original_id"]].copy()
    predictions["original_pred"] = 1
    predictions["perturbed_pred"] = [0, 1, 0]
    return pairs, linkage, predictions


@pytest.mark.parametrize(
    "local_test",
    [
        "test_actual_saved_population_and_historical_artifacts_are_preserved",
        "test_current_prediction_reduction_accepts_csv_rounding_and_refuses_changed_counts",
    ],
)
def test_private_saved_checks_skip_when_checkout_has_no_private_inputs(
    local_test: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    with pytest.raises(pytest.skip.Exception, match="private"):
        globals()[local_test]()


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
    pairs, linkage, predictions = _load_private_saved_inputs()
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
            content = (ROOT / name).read_bytes()
            if name == "docs/artifacts/checkpoint_comparison.json":
                report = json.loads(content)
                validation = report.pop("inference_context_check", None)
                if validation is not None:
                    assert validation["identity_and_settings_agree"] is True
                    assert "historical weights are unverified" in validation["scope"]
                    content = (json.dumps(report, indent=2) + "\n").encode()
            assert hashlib.sha256(content).hexdigest() == digest, name


def test_new_server_uses_separate_ports_and_refuses_role_or_cohort_mismatch() -> None:

    assert default_port("reviewer_1") == 8767
    assert default_port("reviewer_2") == 8768
    with pytest.raises(ValueError):
        default_port("outside")


def test_current_prediction_reduction_accepts_csv_rounding_and_refuses_changed_counts() -> None:
    pairs, linkage, predictions = _load_private_saved_inputs()
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


def test_synthetic_prediction_reduction_covers_csv_rounding_and_count_refusal_in_ci() -> None:
    pairs, linkage, predictions = _synthetic_saved_inputs()
    current, _, _ = build_retrospective_cells(
        predictions, linkage, target_id="protectai-v2", excluded_surrogates=["bilstm-attention"]
    )
    reference = pd.read_csv(StringIO(current.to_csv(index=False)))
    assert not reference["source_gated_yield"].equals(current["source_gated_yield"])
    historical, _, _ = build_retrospective_cells(predictions, linkage, target_id="protectai-v2")
    historical = historical.rename(
        columns={
            "target_original_positive_successful": "target_original_injection",
            "verified_flips_source_successful": "true_target_flips",
        }
    )
    manifest = {"n_surrogates": 3, "n_recipes": 1, "attack_attempts_per_cell": 7}
    checked = {"inference_context_check": {"identity_and_settings_agree": True}}
    params = {
        "sample_trials": 3,
        "expected_population": {"attempted_trials": 14, "changed_trials": 7, "unique_pairs": 1},
    }
    result = prepare_flip_cohort(
        pairs, linkage, predictions, reference, historical, manifest, checked, params, 14
    )
    assert result["attempted_trials"] == 14
    reference.loc[0, "verified_flips_all_candidates"] += 1
    with pytest.raises(ValueError, match="reference cells"):
        prepare_flip_cohort(
            pairs, linkage, predictions, reference, historical, manifest, checked, params, 14
        )
