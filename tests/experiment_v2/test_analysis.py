"""Held-out selection, honest adjudication states, and complete attempt denominators."""

from __future__ import annotations

import pytest

from transfer_risk.lib.experiment_v2.analysis import analyse_experiment, freeze_selection
from transfer_risk.lib.experiment_v2.protocol import content_digest
from transfer_risk.lib.experiment_v2.records import validate_records
from transfer_risk.lib.experiment_v2.schemas import TokenizationDiagnostic
from transfer_risk.lib.experiment_v2.selection import utility

from .fixtures import digest, evidence, records_for


def complete_fixture():
    protocol, corpus, registration = evidence()
    calibration = records_for("calibration", protocol, corpus)
    evaluation = records_for("evaluation", protocol, corpus)
    lock = freeze_selection(
        protocol, corpus, registration, calibration, 20260616, "2026-10-03T00:00:00+00:00"
    )
    evaluation = evaluation.model_copy(
        update={
            "selection_lock_sha256": lock["lock_sha256"],
            "selection_registry_uri": "https://example.invalid/synthetic-selection-lock",
            "selection_registry_evidence_sha256": digest("selection_registration"),
            "selection_registered_at": "2026-10-03T01:00:00+00:00",
        }
    )
    return protocol, corpus, registration, calibration, evaluation, lock


def analyse_fixture(parts):
    protocol, corpus, registration, calibration, evaluation, lock = parts
    return analyse_experiment(
        protocol, corpus, registration, calibration, evaluation, lock, 20260616
    )


def test_selection_is_frozen_before_evaluation_and_uses_group_level_heldout_yield():
    parts = complete_fixture()
    lock = parts[-1]
    primary = lock["selections"]["primary:0"]
    assert primary["cka"] == ["source_0", "source_1"]
    assert primary["source_only"] == ["source_7", "source_6"]
    report = analyse_fixture(parts)
    outcome = report["selection"]["primary"]
    assert outcome["status"] == "complete"
    assert outcome["cka_yield"] == 1.0
    assert outcome["source_only_yield"] == 0.25
    assert outcome["random_expected_yield"] == pytest.approx(34 / 112)
    assert outcome["random_subset_count"] == 28
    assert outcome["finite_pool_random_rank_probability"] == pytest.approx(1 / 28)
    assert outcome["evaluation_groups"] == 4
    assert outcome["replicates"] == 3
    assert report["outcomes"]["primary"]["attempted"] == 96
    assert report["outcomes"]["primary"]["human_valid"] == 96
    assert report["outcomes"]["primary"]["raw_target_flips"] == 15
    assert report["outcomes"]["primary"]["valid_target_flips"] == 15
    assert report["application_study"]["status"] == "disabled"
    assert "original_id" not in str(report)
    assert "candidate_sha256" not in str(report)


def test_unreviewed_candidates_produce_bounds_without_validity_or_selection_claims():
    parts = list(complete_fixture())
    parts[4] = parts[4].model_copy(update={"annotations": []})
    report = analyse_fixture(parts)
    outcome = report["outcomes"]["primary"]
    assert outcome["raw_target_flips"] == 15
    assert outcome["human_valid"] == 0
    assert outcome["human_unresolved"] == 96
    assert outcome["valid_target_yield_lower"] == 0.0
    assert outcome["valid_target_yield_upper"] == pytest.approx(15 / 96)
    assert report["selection"]["primary"]["status"] == "incomplete_human_review"
    assert report["selection"]["primary"]["decision"] is None


def test_mechanical_pass_does_not_substitute_for_human_semantic_or_intent_review():
    parts = list(complete_fixture())
    review = parts[4].annotations[0]
    votes = [vote.model_copy(update={"semantic": "changed"}) for vote in review.votes]
    invalid = review.model_copy(update={"votes": votes})
    parts[4] = parts[4].model_copy(update={"annotations": [invalid, *parts[4].annotations[1:]]})
    report = analyse_fixture(parts)
    assert report["outcomes"]["primary"]["human_invalid"] == 1
    assert report["outcomes"]["primary"]["mechanical_passed"] == 96
    assert report["outcomes"]["primary"]["human_valid"] == 95


def test_source_only_baseline_cannot_be_frozen_with_missing_calibration_adjudication():
    protocol, corpus, registration = evidence()
    calibration = records_for("calibration", protocol, corpus).model_copy(
        update={"annotations": []}
    )
    with pytest.raises(ValueError, match=r"calibration.*human"):
        freeze_selection(
            protocol, corpus, registration, calibration, 20260616, "2026-10-03T00:00:00+00:00"
        )


def test_query_cap_and_zero_target_search_feedback_are_enforced():
    parts = list(complete_fixture())
    attempt = parts[4].attempts[0].model_copy(update={"source_queries": 101})
    parts[4] = parts[4].model_copy(update={"attempts": [attempt, *parts[4].attempts[1:]]})
    with pytest.raises(ValueError, match="query budget"):
        analyse_fixture(parts)
    attempt = attempt.model_copy(update={"source_queries": 100, "target_search_queries": 1})
    parts[4] = parts[4].model_copy(update={"attempts": [attempt, *parts[4].attempts[1:]]})
    with pytest.raises(ValueError, match=r"target.*search"):
        analyse_fixture(parts)


def test_missing_attempt_cannot_silently_improve_the_yield_denominator():
    parts = list(complete_fixture())
    parts[4] = parts[4].model_copy(update={"attempts": parts[4].attempts[1:]})
    with pytest.raises(ValueError, match="attempt coverage"):
        analyse_fixture(parts)


def test_target_identity_drift_between_clean_and_perturbed_predictions_is_rejected():
    parts = list(complete_fixture())
    attempt = parts[4].attempts[0]
    drifted = attempt.target_predictions[0].model_copy(
        update={"checkpoint_sha256": digest("different_target")}
    )
    attempt = attempt.model_copy(
        update={"target_predictions": [drifted, *attempt.target_predictions[1:]]}
    )
    parts[4] = parts[4].model_copy(update={"attempts": [attempt, *parts[4].attempts[1:]]})
    with pytest.raises(ValueError, match="identity"):
        analyse_fixture(parts)


def test_recomputed_or_tampered_selection_lock_is_rejected():
    parts = list(complete_fixture())
    parts[-1] = {**parts[-1], "frozen_at": "2026-10-05T00:00:00+00:00"}
    with pytest.raises(ValueError, match="selection lock"):
        analyse_fixture(parts)


def test_duplicate_human_identity_is_not_independent_adjudication():
    parts = list(complete_fixture())
    review = parts[4].annotations[0]
    review = review.model_copy(update={"votes": [review.votes[0], review.votes[0]]})
    parts[4] = parts[4].model_copy(update={"annotations": [review, *parts[4].annotations[1:]]})
    with pytest.raises(ValueError, match="independent"):
        analyse_fixture(parts)


def test_clean_utility_reports_hard_negative_errors_and_eligibility_checks_each_subtype():

    protocol, corpus, registration = evidence()
    protocol = protocol.model_copy(update={"utility_max_fpr": 0.3})
    registration = registration.model_copy(update={"protocol_sha256": content_digest(protocol)})
    calibration = records_for("calibration", protocol, corpus)
    quoted = next(
        row.id
        for row in corpus.records
        if row.partition == "calibration" and row.negative_subtype == "quoted_instruction"
    )
    clean = [
        item.model_copy(update={"raw_label": "1"})
        if item.model_id == "target_a" and item.original_id == quoted
        else item
        for item in calibration.clean_predictions
    ]
    calibration = calibration.model_copy(update={"clean_predictions": clean})
    index = validate_records(protocol, corpus, registration, calibration, 20260616)
    metrics = utility(index, "target_a", 0)
    assert metrics["fpr"] == 0.25
    assert metrics["quoted_instruction_fpr"] == 0.5
    assert metrics["neutral_instruction_fpr"] == 0.0
    with pytest.raises(ValueError, match="utility"):
        freeze_selection(
            protocol, corpus, registration, calibration, 20260616, "2026-10-03T00:00:00+00:00"
        )


def test_skips_and_errors_remain_in_all_attempt_denominators_and_actual_query_totals():
    parts = list(complete_fixture())
    evaluation = parts[4]
    first, second = evaluation.attempts[:2]
    clean = [
        item.model_copy(update={"raw_label": "0"})
        if (item.model_id, item.replicate, item.original_id)
        == (first.source_id, first.replicate, first.original_id)
        else item
        for item in evaluation.clean_predictions
    ]
    absent = {
        "candidate_sha256": None,
        "source_raw_perturbed": None,
        "target_predictions": [],
        "mechanical_screen": "unassessable",
        "mechanical_evidence_sha256": None,
    }
    skipped = first.model_copy(
        update={**absent, "status": "skipped_source_benign", "source_queries": 1}
    )
    failed = second.model_copy(update={**absent, "status": "error", "source_queries": 100})
    parts[4] = evaluation.model_copy(
        update={
            "clean_predictions": clean,
            "attempts": [skipped, failed, *evaluation.attempts[2:]],
            "annotations": evaluation.annotations[2:],
        }
    )
    result = analyse_fixture(parts)["outcomes"]["primary"]
    assert result["attempted"] == 96
    assert result["returned_candidates"] == 94
    assert result["source_original_positive"] == 95
    assert result["technical_errors"] == 1
    assert result["skipped_source_benign"] == 1
    assert result["raw_target_flips"] == 13
    assert result["source_queries"] == 4801
    assert result["source_query_cap_total"] == 9600


def test_target_benign_original_is_a_baseline_error_and_zero_primary_policy_yield():
    parts = list(complete_fixture())
    evaluation = parts[4]
    clean = [
        item.model_copy(update={"raw_label": "0"})
        if item.model_id == "target_a" and item.original_id == "evaluation_1_0"
        else item
        for item in evaluation.clean_predictions
    ]
    parts[4] = evaluation.model_copy(update={"clean_predictions": clean})
    report = analyse_fixture(parts)
    assert report["outcomes"]["primary"]["target_original_positive"] == 72
    assert report["outcomes"]["primary"]["raw_target_flips"] == 9
    assert report["selection"]["primary"]["cka_yield"] == pytest.approx(3 / 4)
    assert report["selection"]["primary"]["evaluation_positive_originals"] == 4


def test_pre_query_setup_error_retains_zero_actual_queries_without_inventing_a_baseline_call():
    parts = list(complete_fixture())
    evaluation = parts[4]
    attempt = evaluation.attempts[0].model_copy(
        update={
            "status": "error",
            "source_queries": 0,
            "candidate_sha256": None,
            "source_raw_perturbed": None,
            "target_predictions": [],
            "mechanical_screen": "unassessable",
            "mechanical_evidence_sha256": None,
        }
    )
    parts[4] = evaluation.model_copy(
        update={
            "attempts": [attempt, *evaluation.attempts[1:]],
            "annotations": evaluation.annotations[1:],
        }
    )
    report = analyse_fixture(parts)
    assert report["outcomes"]["primary"]["attempted"] == 96
    assert report["outcomes"]["primary"]["technical_errors"] == 1
    assert report["outcomes"]["primary"]["source_queries"] == 4750


def test_unlabeled_nonflips_do_not_block_selector_freezing_or_outcome_inference():
    parts = list(complete_fixture())
    calibration = parts[3]
    source_flips = {attempt.id for attempt in calibration.attempts if attempt.status == "success"}
    calibration = calibration.model_copy(
        update={
            "annotations": [
                row for row in calibration.annotations if row.attempt_id in source_flips
            ]
        }
    )
    lock = freeze_selection(
        parts[0], parts[1], parts[2], calibration, 20260616, "2026-10-03T00:00:00+00:00"
    )
    evaluation = parts[4]
    target_flips = {
        attempt.id
        for attempt in evaluation.attempts
        if any(prediction.raw_label == "0" for prediction in attempt.target_predictions)
    }
    evaluation = evaluation.model_copy(
        update={
            "selection_lock_sha256": lock["lock_sha256"],
            "annotations": [
                row for row in evaluation.annotations if row.attempt_id in target_flips
            ],
        }
    )
    parts[3], parts[4], parts[5] = calibration, evaluation, lock
    report = analyse_fixture(parts)
    assert report["outcomes"]["primary"]["human_unresolved"] == 81
    assert report["selection"]["primary"]["status"] == "complete"
    assert report["selection"]["primary"]["decision"] is not None
    assert report["selection"]["primary"]["outcome_review"]["unresolved_pairs"] == 0
    assert report["human_review"]["evaluation"]["full_review_complete"] is False


def test_unlabeled_possible_flip_still_blocks_confirmed_selector_outcomes():
    parts = list(complete_fixture())
    missing = parts[4].annotations[0].attempt_id
    parts[4] = parts[4].model_copy(
        update={"annotations": [row for row in parts[4].annotations if row.attempt_id != missing]}
    )
    result = analyse_fixture(parts)["selection"]["primary"]
    assert result["status"] == "incomplete_human_review"
    assert result["decision"] is None
    assert result["outcome_review"]["unresolved_pairs"] == 1
    assert result["cka_yield_upper"] > result["cka_yield_lower"]


def test_joint_transitions_keep_logically_valid_and_unresolved_counts_separate():
    parts = list(complete_fixture())
    evaluation = parts[4]
    joint = next(
        attempt
        for attempt in evaluation.attempts
        if attempt.status == "success"
        and any(item.raw_label == "0" for item in attempt.target_predictions)
    )
    review = next(row for row in evaluation.annotations if row.attempt_id == joint.id)
    unresolved = review.model_copy(
        update={"votes": [vote.model_copy(update={"intent": "uncertain"}) for vote in review.votes]}
    )
    parts[4] = evaluation.model_copy(
        update={
            "annotations": [
                unresolved if row.attempt_id == joint.id else row for row in evaluation.annotations
            ]
        }
    )
    result = analyse_fixture(parts)["outcomes"]["primary"]
    assert result["joint_source_target_flips"] == 3
    assert result["valid_joint_source_target_flips"] == 2
    assert result["unresolved_joint_source_target_flips"] == 1


def test_truncation_report_preserves_attempt_denominators_and_realized_seed_scope():
    parts = list(complete_fixture())
    evaluation = parts[4]

    diagnostic = TokenizationDiagnostic(input_tokens=600, max_tokens=512, truncated=True)
    clean = evaluation.clean_predictions[0].model_copy(update={"tokenization": diagnostic})
    attempt = evaluation.attempts[0]
    prediction = attempt.target_predictions[0].model_copy(update={"tokenization": diagnostic})
    attempt = attempt.model_copy(
        update={
            "source_candidate_tokenization": diagnostic,
            "target_predictions": [prediction, *attempt.target_predictions[1:]],
        }
    )
    parts[4] = evaluation.model_copy(
        update={
            "clean_predictions": [clean, *evaluation.clean_predictions[1:]],
            "attempts": [attempt, *evaluation.attempts[1:]],
        }
    )
    report = analyse_fixture(parts)
    diagnostics = report["tokenization_diagnostics"]
    assert diagnostics["planned_attempts"] == len(evaluation.attempts)
    assert diagnostics["models"][clean.model_id]["originals"]["truncated"] == 1
    assert diagnostics["models"][prediction.model_id]["candidates"]["truncated"] == 1
    assert "three realized training replicates" in report["inference_scope"]
