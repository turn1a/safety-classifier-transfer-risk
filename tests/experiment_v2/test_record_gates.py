"""Rerun accounting rejects leakage, identity changes, missing controls, and fake reviews."""

from dataclasses import replace

import pytest

from transfer_risk.lib.experiment_v2 import records as record_contract
from transfer_risk.lib.experiment_v2.protocol import validate_preflight
from transfer_risk.lib.experiment_v2.records import human_state, validate_records
from transfer_risk.lib.experiment_v2.selection import group_vector

from .fixtures import digest, evidence, records_for


@pytest.mark.parametrize("field", ["task_context_sha256", "trusted_instruction_sha256"])
def test_pure_injection_label_requires_context_and_a_trust_boundary(field):
    protocol, corpus, registration = evidence()
    row = corpus.records[-1].model_copy(update={field: None})
    corpus = corpus.model_copy(update={"records": [*corpus.records[:-1], row]})
    report = validate_preflight(protocol, corpus, registration)
    assert any("context" in issue or "trust" in issue for issue in report["issues"])


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ("missing_clean", "clean prediction coverage"),
        ("missing_cka", "similarity coverage"),
        ("wrong_probe", "probe partition"),
        ("nonfinite_cka", "finite"),
        ("missing_realization", "realization coverage"),
        ("wrong_training_seed", "wrong seed"),
        ("wrong_training_split", "train partition"),
        ("bad_execution_ref", "execution_ref"),
        ("changed_frozen_weights", "frozen model identity"),
        ("self_target_weights", "identical target"),
        ("wrong_source_identity", "source attack identity"),
        ("wrong_attack_seed", "attack cell seed"),
        ("duplicate_attempt", "attempt coverage"),
        ("missing_target", "target candidate prediction coverage"),
        ("false_source_success", "actual classifier transition"),
        ("wrong_human_pair", "human annotation pair"),
        ("duplicate_human_pair", "human annotation coverage"),
        ("unmapped_raw_label", "unmapped raw label"),
        ("phase_before_registration", "precedes prospective"),
        ("completed_before_started", "completion precedes"),
    ],
)
def test_invalid_supplied_evidence_is_rejected_before_yield_analysis(mutation, pattern):
    protocol, corpus, registration = evidence()
    records = records_for("calibration", protocol, corpus)
    if mutation == "missing_clean":
        records = records.model_copy(update={"clean_predictions": records.clean_predictions[1:]})
    elif mutation == "missing_cka":
        records = records.model_copy(update={"similarities": records.similarities[1:]})
    elif mutation in {"wrong_probe", "nonfinite_cka"}:
        update = (
            {"probe_partition_sha256": digest("wrong_partition")}
            if mutation == "wrong_probe"
            else {"mean_cka": float("nan")}
        )
        value = records.similarities[0].model_copy(update=update)
        records = records.model_copy(update={"similarities": [value, *records.similarities[1:]]})
    elif mutation == "missing_realization":
        records = records.model_copy(update={"realizations": records.realizations[1:]})
    elif mutation in {
        "wrong_training_seed",
        "wrong_training_split",
        "bad_execution_ref",
        "changed_frozen_weights",
        "self_target_weights",
    }:
        position = (
            6
            if mutation in {"wrong_training_seed", "wrong_training_split", "self_target_weights"}
            else 0
        )
        value = records.realizations[position]
        updates = {
            "wrong_training_seed": {"training_seed": 1},
            "wrong_training_split": {"train_partition_sha256": digest("wrong_split")},
            "bad_execution_ref": {"execution_ref": "f" * 40},
            "changed_frozen_weights": {"checkpoint_sha256": digest("new_target")},
            "self_target_weights": {"checkpoint_sha256": records.realizations[0].checkpoint_sha256},
        }
        realizations = list(records.realizations)
        realizations[position] = value.model_copy(update=updates[mutation])
        records = records.model_copy(update={"realizations": realizations})
    elif mutation in {
        "wrong_source_identity",
        "wrong_attack_seed",
        "missing_target",
        "false_source_success",
    }:
        attempt = records.attempts[0]
        updates = {
            "wrong_source_identity": {"source_checkpoint_sha256": digest("other_source")},
            "wrong_attack_seed": {"attack_seed": 1},
            "missing_target": {"target_predictions": attempt.target_predictions[1:]},
            "false_source_success": {"status": "success"},
        }
        attempt = attempt.model_copy(update=updates[mutation])
        records = records.model_copy(update={"attempts": [attempt, *records.attempts[1:]]})
    elif mutation == "duplicate_attempt":
        records = records.model_copy(update={"attempts": [*records.attempts, records.attempts[0]]})
    elif mutation == "wrong_human_pair":
        review = records.annotations[0].model_copy(
            update={"candidate_sha256": digest("other_candidate")}
        )
        records = records.model_copy(update={"annotations": [review, *records.annotations[1:]]})
    elif mutation == "duplicate_human_pair":
        records = records.model_copy(
            update={"annotations": [*records.annotations, records.annotations[0]]}
        )
    elif mutation == "unmapped_raw_label":
        prediction = records.clean_predictions[0].model_copy(update={"raw_label": "2"})
        records = records.model_copy(
            update={"clean_predictions": [prediction, *records.clean_predictions[1:]]}
        )
    elif mutation == "phase_before_registration":
        records = records.model_copy(update={"started_at": "2026-09-01T00:00:00+00:00"})
    elif mutation == "completed_before_started":
        records = records.model_copy(update={"completed_at": "2026-09-01T00:00:00+00:00"})
    with pytest.raises(ValueError, match=pattern):
        validate_records(protocol, corpus, registration, records, 20260616)


def test_human_disagreement_requires_an_independent_adjudicator_and_preserves_uncertainty():
    protocol, corpus, _ = evidence()
    review = records_for("calibration", protocol, corpus).annotations[0]
    second = review.votes[1].model_copy(update={"intent": "uncertain"})
    disputed = review.model_copy(update={"votes": [review.votes[0], second]})
    assert human_state(disputed) == "unresolved"
    adjudicator = second.model_copy(update={"reviewer_id": "human_three", "intent": "lost"})
    assert human_state(disputed.model_copy(update={"adjudicator": adjudicator})) == "invalid"
    adjudicator = adjudicator.model_copy(update={"intent": "retained"})
    assert human_state(disputed.model_copy(update={"adjudicator": adjudicator})) == "valid"
    with pytest.raises(ValueError, match="independent"):
        human_state(disputed.model_copy(update={"adjudicator": second}))
    assert human_state(review.model_copy(update={"votes": []})) == "unresolved"


@pytest.mark.parametrize(
    "collection,index,changes,pattern",
    [
        ("realizations", 0, {"checkpoint_sha256": "short"}, "identity requires"),
        ("realizations", 0, {"training_seed": 1}, "cannot claim v2 training"),
        (
            "clean_predictions",
            0,
            {"checkpoint_sha256": digest("other_weights")},
            "clean prediction coverage or identity",
        ),
        (
            "similarities",
            0,
            {"source_checkpoint_sha256": digest("other_weights")},
            "similarity model/tokenizer identity",
        ),
        ("attempts", 0, {"mechanical_evidence_sha256": None}, "mechanical screening"),
        ("attempts", 0, {"candidate_sha256": "short"}, "returned candidate"),
        ("attempts", 0, {"source_raw_perturbed": None}, "returned candidate"),
        ("attempts", 0, {"status": "error"}, "inconsistent source status"),
        ("annotations", 0, {"blinded": False}, "blinded"),
    ],
)
def test_boundary_field_corruption_cannot_enter_validated_indices(
    collection, index, changes, pattern
):
    protocol, corpus, registration = evidence()
    records = records_for("calibration", protocol, corpus)
    items = list(getattr(records, collection))
    items[index] = items[index].model_copy(update=changes)
    records = records.model_copy(update={collection: items})
    with pytest.raises(ValueError, match=pattern):
        validate_records(protocol, corpus, registration, records, 20260616)


@pytest.mark.parametrize(
    "changes,pattern",
    [
        ({"hardware": ""}, "hardware"),
        ({"protocol_sha256": digest("wrong_protocol")}, "protocol, corpus"),
        ({"started_at": "2026-10-02T00:00:00"}, "timezone"),
    ],
)
def test_phase_envelope_requires_realized_provenance_and_timezone(changes, pattern):
    protocol, corpus, registration = evidence()
    records = records_for("calibration", protocol, corpus).model_copy(update=changes)
    with pytest.raises(ValueError, match=pattern):
        validate_records(protocol, corpus, registration, records, 20260616)


def test_group_yield_does_not_weight_a_repeated_family_as_independent_originals():

    protocol, corpus, registration = evidence()
    records = records_for("evaluation", protocol, corpus)
    index = validate_records(protocol, corpus, registration, records, 20260616)
    originals = {
        name: row.model_copy(
            update={
                "group_id": "a"
                if name in {"evaluation_1_0", "evaluation_1_1", "evaluation_1_2"}
                else "b"
            }
        )
        for name, row in index.originals.items()
        if row.label == 1
    }
    grouped = replace(index, originals=originals)
    assert group_vector(grouped, {"evaluation_1_0", "evaluation_1_1", "evaluation_1_2"}) == [
        1.0,
        0.0,
    ]


def test_consensus_uncertainty_stays_unresolved_and_noninjection_is_invalid():
    protocol, corpus, _ = evidence()
    review = records_for("calibration", protocol, corpus).annotations[0]
    uncertain = [vote.model_copy(update={"semantic": "uncertain"}) for vote in review.votes]
    assert human_state(review.model_copy(update={"votes": uncertain})) == "unresolved"
    invalid = [vote.model_copy(update={"is_prompt_injection": False}) for vote in review.votes]
    assert human_state(review.model_copy(update={"votes": invalid})) == "invalid"


@pytest.mark.parametrize("reviewer", [" ", "reviewer-with-trailing-space "])
def test_candidate_review_requires_canonical_nonempty_reviewer_identities(reviewer):
    protocol, corpus, _ = evidence()
    review = records_for("calibration", protocol, corpus).annotations[0]
    vote = review.votes[0].model_copy(update={"reviewer_id": reviewer})
    with pytest.raises(ValueError, match="independent"):
        human_state(review.model_copy(update={"votes": [vote, review.votes[1]]}))


@pytest.mark.parametrize(
    "criterion,value,other",
    [
        ("semantic", "changed", {"intent": "uncertain"}),
        ("intent", "lost", {"semantic": "uncertain"}),
        ("is_prompt_injection", False, {"semantic": "uncertain"}),
    ],
)
def test_unanimous_required_criterion_failure_resolves_validity_without_full_review(
    criterion, value, other
):
    protocol, corpus, _ = evidence()
    review = records_for("calibration", protocol, corpus).annotations[0]
    first, second = [vote.model_copy(update={criterion: value}) for vote in review.votes]
    second = second.model_copy(update=other)
    partial = review.model_copy(update={"votes": [first, second]})
    assert human_state(partial) == "invalid"
    assert record_contract.human_review_complete(partial) is False
    assert human_state(partial.model_copy(update={"votes": [first]})) == "unresolved"


def test_one_exact_pair_adjudication_covers_matching_candidates_across_seed_cells():
    protocol, corpus, registration = evidence()
    records = records_for("evaluation", protocol, corpus)
    first = records.attempts[0]
    repeated = next(
        attempt
        for attempt in records.attempts
        if attempt.original_id == first.original_id
        and attempt.replicate == 1
        and attempt.source_id == first.source_id
    )
    duplicate = repeated.model_copy(update={"candidate_sha256": first.candidate_sha256})
    records = records.model_copy(
        update={
            "attempts": [duplicate if row.id == repeated.id else row for row in records.attempts],
            "annotations": [row for row in records.annotations if row.attempt_id != repeated.id],
        }
    )
    index = validate_records(protocol, corpus, registration, records, 20260616)
    assert index.human[first.id] == "valid"
    assert index.human[repeated.id] == "valid"


def test_exact_pair_adjudications_cannot_disagree_across_duplicate_seed_cells():
    protocol, corpus, registration = evidence()
    records = records_for("evaluation", protocol, corpus)
    first = records.attempts[0]
    repeated = next(
        attempt
        for attempt in records.attempts
        if attempt.original_id == first.original_id
        and attempt.replicate == 1
        and attempt.source_id == first.source_id
    )
    duplicate = repeated.model_copy(update={"candidate_sha256": first.candidate_sha256})
    review = next(row for row in records.annotations if row.attempt_id == repeated.id)
    contradictory = review.model_copy(
        update={
            "candidate_sha256": first.candidate_sha256,
            "votes": [vote.model_copy(update={"intent": "lost"}) for vote in review.votes],
        }
    )
    records = records.model_copy(
        update={
            "attempts": [duplicate if row.id == repeated.id else row for row in records.attempts],
            "annotations": [
                contradictory if row.attempt_id == repeated.id else row
                for row in records.annotations
            ],
        }
    )
    with pytest.raises(ValueError, match=r"exact pair.*contradictory"):
        validate_records(protocol, corpus, registration, records, 20260616)
