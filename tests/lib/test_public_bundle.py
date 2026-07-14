"""Tests for public reporting bundle helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transfer_risk.lib import public_bundle
from transfer_risk.lib.public_bundle import (
    apply_corrected_dbs,
    assign_similarity_group,
    build_model_validation_summary,
    build_public_run_metrics,
    build_redacted_qualitative_audit,
    build_results_manifest,
    build_surrogate_summary,
    corrected_dbs_by_surrogate,
    reconstruct_n_target_benign,
    validate_redacted_qualitative_audit,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PUBLIC_ARTIFACTS = _PROJECT_ROOT / "docs" / "artifacts"
_FORBIDDEN_MANIFEST_FRAGMENTS = (
    "/Users/",
    "s3://",
    "arn:aws",
    "terraform.tfstate",
    "instance_id",
    "bucket",
)


def _tiny_cka_matrices() -> dict[str, np.ndarray]:
    """Return two surrogate CKA matrices for unit tests."""
    return {
        "alpha": np.array([[1.0, 0.2], [0.3, 0.8]]),
        "beta": np.array([[0.9, 0.1], [0.4, 0.7]]),
    }


def _master_frame() -> pd.DataFrame:
    """Return a minimal master table with two surrogates and two recipes."""
    return pd.DataFrame(
        [
            {
                "surrogate": "alpha",
                "recipe": "bae",
                "n_successful": 10,
                "transfer_rate": 0.2,
                "mean_cka": 0.8,
                "dbs": 0.1,
                "target": "target/model",
            },
            {
                "surrogate": "alpha",
                "recipe": "pwws",
                "n_successful": 10,
                "transfer_rate": 0.4,
                "mean_cka": 0.8,
                "dbs": 0.1,
                "target": "target/model",
            },
            {
                "surrogate": "beta",
                "recipe": "bae",
                "n_successful": 20,
                "transfer_rate": 0.25,
                "mean_cka": 0.3,
                "dbs": 0.9,
                "target": "target/model",
            },
            {
                "surrogate": "beta",
                "recipe": "pwws",
                "n_successful": 20,
                "transfer_rate": 0.75,
                "mean_cka": 0.3,
                "dbs": 0.9,
                "target": "target/model",
            },
        ]
    )


def test_reconstruct_n_target_benign_rounds_to_integer() -> None:
    series = reconstruct_n_target_benign(pd.Series([10, 7]), pd.Series([0.3333333333, 0.5]))
    assert series.tolist() == [3, 4]


def test_corrected_dbs_by_surrogate_uses_diagonal_box_similarity() -> None:
    values = corrected_dbs_by_surrogate(_tiny_cka_matrices(), box=0)
    assert values["alpha"] == pytest.approx(0.9)
    assert values["beta"] == pytest.approx(0.8)


def test_apply_corrected_dbs_overwrites_dbs_and_adds_target_benign() -> None:
    corrected = apply_corrected_dbs(_master_frame(), _tiny_cka_matrices(), box=0)
    assert corrected["dbs"].tolist() == pytest.approx([0.9, 0.9, 0.8, 0.8])
    assert corrected["n_target_benign"].tolist() == [2, 4, 5, 15]


def test_assign_similarity_group_maps_m1_m2_and_middle() -> None:
    selection = {"M1": ["alpha"], "M2": ["beta"]}
    assert assign_similarity_group("alpha", selection) == "M1"
    assert assign_similarity_group("beta", selection) == "M2"
    assert assign_similarity_group("gamma", selection) == "middle"


def test_build_surrogate_summary_aggregates_per_surrogate() -> None:
    selection = {"M1": ["alpha"], "M2": ["beta"]}
    summary = build_surrogate_summary(_master_frame(), selection)
    assert list(summary.columns) == [
        "surrogate",
        "mean_cka",
        "dbs",
        "macro_mean_transfer",
        "macro_min_transfer",
        "macro_max_transfer",
        "total_successful",
        "total_target_benign",
        "pooled_target_benign_rate",
        "similarity_group",
    ]
    alpha = summary.loc[summary["surrogate"] == "alpha"].iloc[0]
    assert alpha["macro_mean_transfer"] == pytest.approx(0.3)
    assert alpha["total_successful"] == 20
    assert alpha["total_target_benign"] == 6
    assert alpha["similarity_group"] == "M1"


def test_build_public_run_metrics_includes_corrected_associations() -> None:
    master = pd.read_csv(_PUBLIC_ARTIFACTS / "master_results_table.csv")
    frozen = json.loads((_PUBLIC_ARTIFACTS / "run_metrics.json").read_text())
    metrics = build_public_run_metrics(master, frozen["ablation"], frozen["thresholds"])
    assert metrics["counts"]["n_cells"] == 50
    assert metrics["counts"]["n_surrogates"] == 10
    assert metrics["counts"]["n_recipes"] == 5
    assert metrics["transfer"]["macro_cell_mean"] == pytest.approx(0.38216487917614733)
    assert metrics["transfer"]["pooled_target_benign_rate"] == pytest.approx(0.3204951856946355)
    assert metrics["transfer"]["pooled_target_benign"] == 1398
    assert metrics["transfer"]["pooled_successful"] == 4362
    assert metrics["associations"]["surrogate"]["mean_cka"]["rho"] == pytest.approx(0.7575757576)
    assert metrics["associations"]["surrogate"]["mean_cka"]["two_sided_p"] == pytest.approx(
        0.0149294533
    )
    assert metrics["associations"]["surrogate"]["dbs"]["rho"] == pytest.approx(0.5272727273)
    assert metrics["associations"]["surrogate"]["dbs"]["exact"] is True
    assert metrics["ablation"]["mean_diff_pp"] == pytest.approx(32.3344773)
    assert metrics["ablation"]["mean_p_value"] == pytest.approx(0.10)


def test_summarize_attack_outcomes_counts_only_result_types() -> None:
    adversarial_examples = {
        "alpha__bae": [
            {"result_type": "SuccessfulAttackResult", "original": "private text"},
            {"result_type": "FailedAttackResult", "original": "private text"},
            {"result_type": "SkippedAttackResult", "original": "private text"},
        ],
        "beta__pwws": [
            {"result_type": "SuccessfulAttackResult", "perturbed": "private text"},
            {"result_type": "FailedAttackResult", "perturbed": "private text"},
        ],
    }
    outcomes = public_bundle.summarize_attack_outcomes(adversarial_examples)
    assert outcomes == {
        "attempted": 5,
        "eligible": 4,
        "successful": 2,
        "failed": 2,
        "skipped": 1,
        "eligible_attack_success_rate": 0.5,
    }
    assert "private text" not in str(outcomes)


def test_public_run_metrics_contains_frozen_attack_funnel() -> None:
    metrics = json.loads((_PUBLIC_ARTIFACTS / "run_metrics.json").read_text())
    assert metrics["attack_outcomes"] == {
        "attempted": 9550,
        "eligible": 8600,
        "successful": 4362,
        "failed": 4238,
        "skipped": 950,
        "eligible_attack_success_rate": pytest.approx(4362 / 8600),
    }


def test_build_results_manifest_excludes_forbidden_identifiers() -> None:
    manifest = build_results_manifest(
        {
            "schema_version": "1.0",
            "experiment_commit": "5af7330",
            "root_seed": 20260616,
            "uv_lock_sha256": "fed6bd35da2963e19ac5fe00ab14fff0937120bb2c92e132741256cfc1de8f36",
            "target": "protectai/deberta-v3-base-prompt-injection-v2",
            "probe_window": {"n_probe": 2000, "max_seq_len": 512},
            "attack_window": {"eval_set_size": 191, "max_seq_len": 512, "query_budget": 6000},
            "transfer_window": {"max_seq_len": 512},
            "attack_attempts_per_cell": 191,
            "query_budget": 6000,
            "n_surrogates": 10,
            "n_recipes": 5,
            "hardware": {
                "instance_type": "r8g.48xlarge",
                "vcpus": 192,
                "architecture": "Graviton4",
                "pricing_model": "spot",
                "victim_backend": "torch on ARM",
            },
            "completion_note": (
                "Most attack shards ran on cloud spot; final shard reductions completed locally."
            ),
        }
    )
    encoded = str(manifest).lower()
    for fragment in _FORBIDDEN_MANIFEST_FRAGMENTS:
        assert fragment.lower() not in encoded
    assert manifest["experiment_commit"] == "5af7330"
    assert manifest["root_seed"] == 20260616


def test_build_redacted_qualitative_audit_releases_only_safe_fields() -> None:
    """Release fixed audit labels without source or perturbed text."""
    audit = build_redacted_qualitative_audit()

    assert len(audit) == 2
    assert all(
        set(record)
        == {
            "surrogate",
            "recipe",
            "n_words_changed",
            "label",
            "change_summary",
            "audit_note",
        }
        for record in audit
    )
    assert {record["label"] for record in audit} == {
        "semantic_preservation_uncertain",
        "meaning_changed",
    }
    assert all(
        "verified target prediction change" not in record["audit_note"].lower() for record in audit
    )


def test_validate_redacted_qualitative_audit_rejects_source_text_fields() -> None:
    """Reject a would-be public record containing a sensitive text field."""
    unsafe = [
        {
            "surrogate": "alpha",
            "recipe": "bae",
            "n_words_changed": 1,
            "label": "meaning_changed",
            "change_summary": "Lexical substitution reviewed without retained text.",
            "audit_note": "Historical conditional audit only.",
            "original": "sensitive-marker",
        }
    ]

    with pytest.raises(ValueError, match="forbidden"):
        validate_redacted_qualitative_audit(unsafe)


def test_validate_redacted_qualitative_audit_rejects_missing_safe_field() -> None:
    """Require every fixed safe field before a qualitative record is published."""
    records = build_redacted_qualitative_audit()
    del records[0]["audit_note"]

    with pytest.raises(ValueError, match="fields must match"):
        validate_redacted_qualitative_audit(records)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n_words_changed", "one", "must be an integer"),
        ("change_summary", 1, "string fields must be strings"),
    ],
)
def test_validate_redacted_qualitative_audit_rejects_wrong_safe_field_types(
    field: str,
    value: int | str,
    message: str,
) -> None:
    """Reject noncanonical value types before checking fixed-record equality."""
    records = build_redacted_qualitative_audit()
    records[0][field] = value

    with pytest.raises(ValueError, match=message):
        validate_redacted_qualitative_audit(records)


def test_checked_in_qualitative_artifact_matches_redacted_audit_schema() -> None:
    """Keep the published qualitative JSON equal to the safe redacted release."""
    published = json.loads((_PUBLIC_ARTIFACTS / "qualitative_examples.json").read_text())

    validate_redacted_qualitative_audit(published)
    assert published == build_redacted_qualitative_audit()


def test_build_model_validation_summary_merges_registry_and_metadata() -> None:
    registry = {
        "target": "protectai/deberta-v3-base-prompt-injection-v2",
        "surrogates": [{"name": "bert-base-ft", "kind": "finetune"}],
    }
    metadata = {
        "bert-base-ft": {
            "kind": "finetune",
            "val_accuracy": 0.97,
            "num_params": 110_000_000,
        }
    }
    summary = build_model_validation_summary(registry, metadata)
    assert summary["target"] == registry["target"]
    model = summary["models"][0]
    assert model["name"] == "bert-base-ft"
    assert model["kind"] == "finetune"
    assert model["val_accuracy"] == pytest.approx(0.97)
    assert model["num_params"] == 110_000_000


def test_publish_dataset_audit_preserves_safe_counts_and_documents_scope() -> None:
    audit = {
        "n_raw": 12,
        "n_duplicates_removed": 2,
        "n_final": 10,
        "per_source": {"source-a": 6, "source-b": 4},
    }
    published = public_bundle.publish_dataset_audit(audit)
    assert published["n_raw"] == 12
    assert published["n_final"] == 10
    assert published["per_source"] == {"source-a": 6, "source-b": 4}
    assert published["per_source_count_stage"] == "post_deduplication"
    assert audit == {
        "n_raw": 12,
        "n_duplicates_removed": 2,
        "n_final": 10,
        "per_source": {"source-a": 6, "source-b": 4},
    }
