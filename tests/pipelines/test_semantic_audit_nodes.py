"""Boundary tests for no-model semantic-review preparation and later human review."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from transfer_risk.pipelines.semantic_audit.nodes import (
    pending_semantic_review,
    prepare_semantic_audit,
    render_semantic_review_tools,
    review_semantic_audit,
    summarize_semantic_valid_flip_yield,
)
from transfer_risk.pipelines.semantic_audit.pipeline import (
    create_pipeline,
    create_review_pipeline,
    create_tools_pipeline,
)


def _inputs() -> tuple[dict[str, list[dict[str, Any]]], dict[str, pd.DataFrame], dict[str, Any]]:
    """Build saved records whose failed/skipped originals still require source mapping."""
    original_a, original_b = "private-alpha", "private-beta"
    attacks = {
        "model-a__recipe-a": [
            {
                "original": original_a,
                "perturbed": "private-alpha-change",
                "success": True,
                "original_label": 1,
                "result_type": "SuccessfulAttackResult",
            },
            {
                "original": original_b,
                "perturbed": original_b,
                "success": False,
                "original_label": 1,
                "result_type": "SkippedAttackResult",
            },
        ],
        "model-b__recipe-a": [
            {
                "original": original_a,
                "perturbed": "private-alpha-change",
                "success": False,
                "original_label": 1,
                "result_type": "FailedAttackResult",
            },
        ],
    }
    splits = {
        "test": pd.DataFrame(
            {
                "text": [original_a, original_b, "private-benign"],
                "label": [1, 1, 0],
                "source": ["source-a", "source-b", "source-c"],
            }
        )
    }
    params = {
        "eval_set_size": 2,
        "max_prompt_chars": 100,
        "pairs_per_stratum": 1,
        "reviewer_ids": [],
    }
    return attacks, splits, params


def test_preparation_keeps_every_outcome_private_and_publishes_only_fidelity_aggregates() -> None:
    attacks, splits, params = _inputs()
    pairs, candidates, linkage, summary = prepare_semantic_audit(attacks, splits, params, 42)

    assert len(pairs) == 2
    assert len(candidates) == 2
    assert pairs["original_id"].nunique() == 2
    assert linkage["attempt_count"].sum() == 3
    assert linkage["source_successful_count"].sum() == 1
    assert linkage["source_failed_count"].sum() == 1
    assert linkage["source_skipped_count"].sum() == 1
    assert set(linkage["source"]) == {"source-a", "source-b"}
    assert summary["population"]["attempts"] == 3
    assert summary["population"]["unique_pairs"] == 2
    assert "private-alpha" not in json.dumps(summary)
    assert "private-beta" not in json.dumps(summary)
    assert not set(candidates.columns) & {
        "source",
        "surrogate",
        "recipe",
        "success",
        "original_label",
        "original_id",
        "result_type",
    }


def test_pending_stage_never_creates_labels_and_csv_roundtrip_retains_pending_state() -> None:
    attacks, splits, params = _inputs()
    pairs, candidates, linkage, _ = prepare_semantic_audit(attacks, splits, params, 42)
    pending, summary = pending_semantic_review(pairs, linkage)
    reloaded_template = pd.read_csv(StringIO(candidates.to_csv(index=False)))
    reviewed, reviewed_summary = review_semantic_audit(pairs, linkage, reloaded_template, params)

    pd.testing.assert_frame_equal(pending, reviewed)
    assert summary == reviewed_summary
    assert summary["status"] == "pending"
    assert summary["finite_population_pair_validity_bounds"] == {"lower": 0.0, "upper": 1.0}
    assert set(pending["semantic_preserved"]) == {""}


def test_review_stage_accepts_real_two_reviewer_rows_without_publishing_identities() -> None:
    attacks, splits, params = _inputs()
    pairs, candidates, linkage, _ = prepare_semantic_audit(attacks, splits, params, 42)
    params["reviewer_ids"] = ["human-a", "human-b"]
    annotations = pd.DataFrame(
        [
            {
                "pair_id": pair_id,
                "reviewer_id": reviewer,
                "independent_review": "yes",
                "semantic_preserved": "yes",
                "intent_preserved": "yes",
                "original_is_injection": "yes",
            }
            for pair_id in candidates["pair_id"]
            for reviewer in params["reviewer_ids"]
        ]
    )
    consensus, summary = review_semantic_audit(pairs, linkage, annotations, params)

    assert set(consensus["status"]) == {"valid"}
    assert summary["resolved_pairs"] == 2
    assert summary["finite_population_pair_validity_bounds"] == {"lower": 1.0, "upper": 1.0}
    assert "human-a" not in json.dumps(summary)
    assert "private-alpha" not in json.dumps(summary)


def test_preparation_and_review_have_separate_outputs_and_no_model_inputs() -> None:
    preparation, review = create_pipeline(), create_review_pipeline()

    assert preparation.inputs() == {
        "adversarial_examples",
        "task_splits",
        "params:semantic_audit",
        "params:seed",
    }
    assert review.inputs() == {
        "semantic_audit_private_pairs",
        "semantic_audit_sample_linkage",
        "semantic_audit_annotations",
        "params:semantic_audit",
        "retrospective_predictions__protectai-v2",
        "params:retrospective_audit.targets.protectai-v2",
        "pub_target_audit_cells",
        "pub_results_manifest",
    }
    assert preparation.all_outputs() == {
        "semantic_audit_private_pairs",
        "semantic_audit_annotation_candidates",
        "semantic_audit_sample_linkage",
        "pub_semantic_fidelity_summary",
        "semantic_audit_pending_consensus",
        "pub_semantic_review_preparation_summary",
    }
    assert review.outputs() == {
        "pub_semantic_review_summary",
        "pub_semantic_valid_flip_yield_summary",
    }
    assert "semantic_audit_review_consensus" in review.all_outputs()
    assert not preparation.all_outputs() & review.outputs()


def test_tool_generation_uses_saved_sample_without_resampling_or_changing_review_outputs() -> None:
    attacks, splits, params = _inputs()
    _, candidates, _, _ = prepare_semantic_audit(attacks, splits, params, 42)
    params["reviewer_ids"] = ["reviewer_1", "reviewer_2"]
    template = (
        Path(__file__).parents[2] / "src/transfer_risk/templates/semantic_review_tool.html"
    ).read_text()
    first, second = render_semantic_review_tools(candidates, template, params)
    pipeline = create_tools_pipeline()

    assert '"reviewer_id":"reviewer_1"' in first
    assert '"reviewer_id":"reviewer_2"' in second
    assert "private-alpha" in first
    assert pipeline.inputs() == {
        "semantic_audit_annotation_candidates",
        "semantic_review_html_template",
        "params:semantic_audit",
    }
    assert pipeline.outputs() == {
        "semantic_audit_review_tool_reviewer_1",
        "semantic_audit_review_tool_reviewer_2",
    }
    assert not pipeline.all_outputs() & create_pipeline().all_outputs()
    assert not pipeline.all_outputs() & create_review_pipeline().all_outputs()


def test_valid_flip_node_uses_saved_reference_labels_and_preserves_pending_validity() -> None:
    attacks, splits, params = _inputs()
    attacks["model-b__recipe-a"].append(
        {
            "original": "private-beta",
            "perturbed": "private-beta",
            "success": False,
            "original_label": 1,
            "result_type": "SkippedAttackResult",
        }
    )
    pairs, _, linkage, _ = prepare_semantic_audit(attacks, splits, params, 42)
    pending, _ = pending_semantic_review(pairs, linkage)
    predictions = pairs[["pair_id", "original_id"]].copy()
    predictions["original_pred"] = 1
    predictions["perturbed_pred"] = (pairs["original"] == pairs["perturbed"]).astype(int)
    summary = summarize_semantic_valid_flip_yield(
        pending,
        predictions,
        linkage,
        {"positive_label": 1, "benign_label": 0, "excluded_surrogates": []},
        pd.DataFrame(
            [
                {
                    "surrogate": "model-a",
                    "recipe": "recipe-a",
                    "source_successful": 1,
                    "target_original_injection": 1,
                    "true_target_flips": 1,
                },
                {
                    "surrogate": "model-b",
                    "recipe": "recipe-a",
                    "source_successful": 0,
                    "target_original_injection": 0,
                    "true_target_flips": 0,
                },
            ]
        ),
        {"attack_attempts_per_cell": 2, "n_surrogates": 2, "n_recipes": 1},
    )

    assert summary["target"] == "protectai-v2"
    assert summary["attempted_trials"] == 4
    assert summary["policies"]["source_success_gated"]["valid_flip_count_bounds"] == {
        "lower": 0,
        "upper": 1,
    }
    assert summary["policies"]["all_returned_candidates"]["valid_flip_count_bounds"] == {
        "lower": 0,
        "upper": 2,
    }
    assert "private-alpha" not in json.dumps(summary)


def test_valid_flip_summary_rechecks_rejected_private_target_predictions() -> None:
    """Human labels must not publish outcomes from a rejected target prediction set."""
    consensus = pd.DataFrame([{"pair_id": "p", "status": "unresolved"}])
    predictions = pd.DataFrame(
        [{"pair_id": "p", "original_id": "o", "original_pred": 1, "perturbed_pred": 0}]
    )
    linkage = pd.DataFrame(
        [
            {
                "pair_id": "p",
                "original_id": "o",
                "source": "source",
                "surrogate": "s",
                "recipe": "r",
                "attempt_count": 1,
                "source_successful_count": 1,
                "source_failed_count": 0,
                "source_skipped_count": 0,
                "source_other_count": 0,
                "selected_for_review": True,
            }
        ]
    )
    historical = pd.DataFrame(
        [
            {
                "surrogate": "s",
                "recipe": "r",
                "source_successful": 1,
                "target_original_injection": 1,
                "true_target_flips": 0,
            }
        ]
    )
    params = {"positive_label": 1, "benign_label": 0, "excluded_surrogates": []}
    manifest = {"attack_attempts_per_cell": 1, "n_surrogates": 1, "n_recipes": 1}
    with pytest.raises(ValueError, match="historical"):
        summarize_semantic_valid_flip_yield(
            consensus, predictions, linkage, params, historical, manifest
        )
    historical.loc[0, "true_target_flips"] = 1
    result = summarize_semantic_valid_flip_yield(
        consensus, predictions, linkage, params, historical, manifest
    )
    assert result["reference_check"]["counts_agree"] is True
    assert result["policies"]["all_returned_candidates"]["valid_flip_count_bounds"] == {
        "lower": 0,
        "upper": 1,
    }
