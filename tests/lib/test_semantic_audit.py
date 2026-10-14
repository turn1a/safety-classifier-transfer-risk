"""Behavioral invariants for private semantic-review preparation."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from transfer_risk.lib import semantic_audit


def _record(
    original: str,
    perturbed: str,
    *,
    success: bool = False,
    result_type: str = "FailedAttackResult",
) -> dict[str, Any]:
    """Build one saved attack record without loading real prompt text."""
    return {
        "original": original,
        "perturbed": perturbed,
        "original_label": 1,
        "success": success,
        "result_type": "SuccessfulAttackResult" if success else result_type,
    }


def _population() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create an overlapping two-stratum population with repeated attempts."""
    attacks = {
        "model-a__recipe-a": [
            _record("private original", "private variant A", success=True),
            _record("private original", "private variant A", success=True),
            _record("private original", "private variant B"),
            _record("private original", "private variant C", result_type="SkippedAttackResult"),
        ],
        "model-b__recipe-a": [
            _record("private original", "private variant A"),
            _record("private original", "private variant D"),
        ],
    }
    return semantic_audit.build_pair_population(attacks, {"private original": "source-a"})


def _sample_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select every pair while retaining the actual sampling contract."""
    pairs, linkage = _population()
    candidates, linkage = semantic_audit.sample_annotation_candidates(
        pairs, linkage, pairs_per_stratum=10, seed=42
    )
    return pairs, candidates, linkage


def _reviews(
    pair_id: str, labels: tuple[str, str, str] = ("yes", "yes", "yes")
) -> list[dict[str, str]]:
    """Provide two explicitly independent synthetic reviewer responses."""
    return [
        {
            "pair_id": pair_id,
            "reviewer_id": reviewer,
            "independent_review": "yes",
            "semantic_preserved": labels[0],
            "intent_preserved": labels[1],
            "original_is_injection": labels[2],
        }
        for reviewer in ("reviewer-a", "reviewer-b")
    ]


@pytest.mark.parametrize(
    ("original", "perturbed", "char_edits", "token_edits"),
    [
        ("", "", 0, 0),
        ("kitten", "sitting", 3, 1),
        ("same words", "same words", 0, 0),
        ("a b", "a  b", 1, 0),
        ("ab", "ba", 2, 1),
        ("café now", "cafe now", 1, 1),
        ("a b c", "a c", 2, 1),
        ("abc", "", 3, 1),
    ],
)
def test_fidelity_counts_exact_codepoint_and_whitespace_token_edits(
    original: str, perturbed: str, char_edits: int, token_edits: int
) -> None:
    stats = semantic_audit.pair_fidelity(original, perturbed)

    assert stats["character_edit_distance"] == char_edits
    assert stats["token_edit_distance"] == token_edits
    assert stats["original_characters"] == len(original)
    assert stats["perturbed_characters"] == len(perturbed)
    assert stats["identical"] is (original == perturbed)


def test_ids_preserve_tuple_boundaries_unicode_and_exact_whitespace() -> None:
    ids = {
        semantic_audit.stable_pair_id("ab", "c"),
        semantic_audit.stable_pair_id("a", "bc"),
        semantic_audit.stable_pair_id("a", "bc "),
        semantic_audit.stable_pair_id("é", "x"),
        semantic_audit.stable_pair_id("e\u0301", "x"),
    }

    assert len(ids) == 5
    assert semantic_audit.stable_pair_id("ab", "c") == semantic_audit.stable_pair_id("ab", "c")


def test_population_deduplicates_pairs_without_dropping_attempts_or_outcomes() -> None:
    pairs, linkage = _population()

    assert len(pairs) == 4
    assert len(linkage) == 5
    assert linkage["attempt_count"].sum() == 6
    assert linkage["source_successful_count"].sum() == 2
    assert linkage["source_failed_count"].sum() == 3
    assert linkage["source_skipped_count"].sum() == 1
    assert pairs["original_id"].nunique() == 1
    assert set(linkage["source"]) == {"source-a"}


def test_population_rejects_unmapped_original_instead_of_dropping_it() -> None:
    with pytest.raises(ValueError, match="source mapping"):
        semantic_audit.build_pair_population(
            {"model-a__recipe-a": [_record("private unmatched original", "private change")]}, {}
        )


def test_population_rejects_status_success_contradictions() -> None:
    record = _record("private original", "private variant")
    record["result_type"] = "SuccessfulAttackResult"

    with pytest.raises(ValueError, match="success"):
        semantic_audit.build_pair_population(
            {"model-a__recipe-a": [record]}, {"private original": "source-a"}
        )


def test_union_sampling_records_exact_inclusion_probabilities_for_shared_pairs() -> None:
    pairs, linkage = _population()
    candidates, sampled_linkage = semantic_audit.sample_annotation_candidates(
        pairs, linkage, pairs_per_stratum=1, seed=42
    )
    probability_by_pair = sampled_linkage.groupby("pair_id")["inclusion_probability"].first()
    pair_a = semantic_audit.stable_pair_id("private original", "private variant A")
    pair_b = semantic_audit.stable_pair_id("private original", "private variant B")
    pair_d = semantic_audit.stable_pair_id("private original", "private variant D")

    assert probability_by_pair[pair_a] == pytest.approx(2 / 3)
    assert probability_by_pair[pair_b] == pytest.approx(1 / 3)
    assert probability_by_pair[pair_d] == pytest.approx(1 / 2)
    assert sampled_linkage.groupby(["source", "surrogate", "recipe"])[
        "drawn_in_stratum"
    ].sum().tolist() == [1, 1]
    assert not candidates["pair_id"].duplicated().any()
    assert set(candidates["pair_id"]) == set(
        sampled_linkage.loc[sampled_linkage["selected_for_review"], "pair_id"]
    )


def test_sampling_is_order_and_outcome_independent_and_annotation_view_is_blinded() -> None:
    pairs, linkage = _population()
    candidates, sampled = semantic_audit.sample_annotation_candidates(
        pairs, linkage, pairs_per_stratum=1, seed=42
    )
    altered_linkage = linkage.iloc[::-1].copy()
    altered_linkage["source_successful_count"] = 0
    altered_linkage["source_failed_count"] = altered_linkage["attempt_count"]
    altered_linkage["source_skipped_count"] = 0
    reordered, sampled_reordered = semantic_audit.sample_annotation_candidates(
        pairs.iloc[::-1], altered_linkage, pairs_per_stratum=1, seed=42
    )

    pd.testing.assert_frame_equal(candidates, reordered)
    assert set(sampled.loc[sampled["selected_for_review"], "pair_id"]) == set(
        sampled_reordered.loc[sampled_reordered["selected_for_review"], "pair_id"]
    )
    assert set(candidates.columns) == {
        "pair_id",
        "original",
        "perturbed",
        "reviewer_id",
        "independent_review",
        "semantic_preserved",
        "intent_preserved",
        "original_is_injection",
        "notes",
        "review_status",
    }
    assert set(candidates["review_status"]) == {"pending"}
    for field in (
        "reviewer_id",
        "independent_review",
        "semantic_preserved",
        "intent_preserved",
        "original_is_injection",
    ):
        assert set(candidates[field]) == {""}


def test_pending_labels_leave_the_entire_population_validity_interval_open() -> None:
    pairs, candidates, linkage = _sample_all()
    consensus = semantic_audit.resolve_annotations(candidates, candidates["pair_id"].tolist(), [])
    summary = semantic_audit.review_summary(pairs, linkage, consensus)

    assert set(consensus["status"]) == {"unresolved"}
    assert summary["confirmed_valid_pairs"] == 0
    assert summary["confirmed_invalid_pairs"] == 0
    assert summary["resolved_pairs"] == 0
    assert summary["finite_population_pair_validity_bounds"] == {"lower": 0.0, "upper": 1.0}
    assert summary["design_weighted_validity_estimate"] is None


@pytest.mark.parametrize(
    "reviewer_ids", [["one"], ["same", " SAME "], ["", "two"], ["one", "two", "three"]]
)
def test_reviewer_configuration_requires_two_distinct_nonempty_identities(
    reviewer_ids: list[str],
) -> None:
    with pytest.raises(ValueError, match="two distinct"):
        semantic_audit.resolve_annotations(pd.DataFrame(), ["pair-a"], reviewer_ids)


def test_one_identity_string_cannot_be_interpreted_as_two_reviewer_characters() -> None:
    with pytest.raises(ValueError, match="two distinct"):
        semantic_audit.resolve_annotations(pd.DataFrame(), ["pair-a"], "ab")


def test_submitted_reviews_require_registered_identities_and_independence_attestation() -> None:
    annotations = pd.DataFrame(_reviews("pair-a"))
    with pytest.raises(ValueError, match="two distinct"):
        semantic_audit.resolve_annotations(annotations, ["pair-a"], [])

    annotations.loc[1, "independent_review"] = "no"
    with pytest.raises(ValueError, match="independent_review"):
        semantic_audit.resolve_annotations(annotations, ["pair-a"], ["reviewer-a", "reviewer-b"])


@pytest.mark.parametrize(
    "field", ["semantic_preserved", "intent_preserved", "original_is_injection"]
)
def test_every_validity_dimension_needs_explicit_agreement(field: str) -> None:
    annotations = pd.DataFrame(_reviews("pair-a"))
    annotations.loc[1, field] = "no"
    consensus = semantic_audit.resolve_annotations(
        annotations, ["pair-a"], ["reviewer-a", "reviewer-b"]
    )
    assert consensus.iloc[0]["status"] == "unresolved"
    assert consensus.iloc[0]["reason"] == "disagreement"

    annotations.loc[0, field] = "no"
    consensus = semantic_audit.resolve_annotations(
        annotations, ["pair-a"], ["reviewer-a", "reviewer-b"]
    )
    assert consensus.iloc[0]["status"] == "invalid"


@pytest.mark.parametrize("label", ["uncertain", "", None])
def test_uncertain_or_incomplete_reviews_stay_unresolved(label: str | None) -> None:
    annotations = pd.DataFrame(_reviews("pair-a"))
    annotations.loc[1, "intent_preserved"] = label
    consensus = semantic_audit.resolve_annotations(
        annotations, ["pair-a"], ["reviewer-a", "reviewer-b"]
    )

    assert consensus.iloc[0]["status"] == "unresolved"


@pytest.mark.parametrize(
    "failed", ["semantic_preserved", "intent_preserved", "original_is_injection"]
)
@pytest.mark.parametrize("other_state", ["uncertain", "missing", "disagreement"])
def test_agreed_no_disproves_validity_even_when_other_criteria_are_unresolved(
    failed: str, other_state: str
) -> None:
    annotations = pd.DataFrame(_reviews("pair-a"))
    annotations[failed] = "no"
    other = next(
        label
        for label in ("semantic_preserved", "intent_preserved", "original_is_injection")
        if label != failed
    )
    if other_state == "uncertain":
        annotations[other] = "uncertain"
    elif other_state == "missing":
        annotations[other] = ""
    else:
        annotations.loc[1, other] = "no"
    consensus = semantic_audit.resolve_annotations(
        annotations, ["pair-a"], ["reviewer-a", "reviewer-b"]
    )

    assert consensus.iloc[0]["status"] == "invalid"
    assert consensus.iloc[0][failed] == "no"
    assert consensus.iloc[0]["reason"] == "agreed_no_criterion"
    assert consensus.iloc[0]["complete_review_count"] == (0 if other_state == "missing" else 2)


def test_one_independent_reviewer_no_cannot_resolve_validity() -> None:
    annotations = pd.DataFrame(_reviews("pair-a", ("no", "uncertain", "yes"))[:1])
    consensus = semantic_audit.resolve_annotations(
        annotations, ["pair-a"], ["reviewer-a", "reviewer-b"]
    )
    assert consensus.iloc[0]["status"] == "unresolved"

    second = pd.DataFrame(_reviews("pair-a", ("no", "uncertain", "yes")))
    second.loc[1, "independent_review"] = ""
    consensus = semantic_audit.resolve_annotations(second, ["pair-a"], ["reviewer-a", "reviewer-b"])
    assert consensus.iloc[0]["status"] == "unresolved"


def test_duplicate_or_unknown_reviewer_and_unknown_pairs_are_rejected() -> None:
    duplicated = pd.DataFrame(_reviews("pair-a")[:1] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        semantic_audit.resolve_annotations(duplicated, ["pair-a"], ["reviewer-a", "reviewer-b"])

    unknown_reviewer = pd.DataFrame(_reviews("pair-a"))
    unknown_reviewer.loc[1, "reviewer_id"] = "unregistered"
    with pytest.raises(ValueError, match="registered"):
        semantic_audit.resolve_annotations(
            unknown_reviewer, ["pair-a"], ["reviewer-a", "reviewer-b"]
        )

    with pytest.raises(ValueError, match="selected"):
        semantic_audit.resolve_annotations(
            pd.DataFrame(_reviews("outside")), ["pair-a"], ["reviewer-a", "reviewer-b"]
        )


def test_review_bounds_use_known_labels_and_preserve_attempt_multiplicity() -> None:
    pairs, candidates, linkage = _sample_all()
    pair_a = semantic_audit.stable_pair_id("private original", "private variant A")
    pair_b = semantic_audit.stable_pair_id("private original", "private variant B")
    annotations = pd.DataFrame([*_reviews(pair_a), *_reviews(pair_b, ("no", "yes", "yes"))])
    consensus = semantic_audit.resolve_annotations(
        annotations, candidates["pair_id"].tolist(), ["reviewer-a", "reviewer-b"]
    )
    summary = semantic_audit.review_summary(pairs, linkage, consensus)

    assert summary["confirmed_valid_pairs"] == 1
    assert summary["confirmed_invalid_pairs"] == 1
    assert summary["unresolved_population_pairs"] == 2
    assert summary["finite_population_pair_validity_bounds"] == {"lower": 0.25, "upper": 0.75}
    assert summary["finite_population_attempt_validity_bounds"]["lower"] == pytest.approx(3 / 6)
    assert summary["finite_population_attempt_validity_bounds"]["upper"] == pytest.approx(5 / 6)
    assert summary["design_weighted_validity_estimate"] is None


def test_complete_reviews_allow_weighted_pair_estimate_without_claiming_cell_rates() -> None:
    pairs, linkage = _population()
    candidates, linkage = semantic_audit.sample_annotation_candidates(
        pairs, linkage, pairs_per_stratum=1, seed=42
    )
    annotations = pd.DataFrame(
        [row for pair_id in candidates["pair_id"] for row in _reviews(pair_id)]
    )
    consensus = semantic_audit.resolve_annotations(
        annotations, candidates["pair_id"].tolist(), ["reviewer-a", "reviewer-b"]
    )
    summary = semantic_audit.review_summary(pairs, linkage, consensus)

    assert summary["design_weighted_validity_estimate"] == 1.0
    assert summary["cell_validity_rates_estimated"] is False
    assert summary["cka_validity_association"] is None
    assert summary["bounds_are_confidence_intervals"] is False


def test_weighted_estimate_uses_union_weights_once_per_selected_pair() -> None:
    pairs, linkage = _population()
    _, linkage = semantic_audit.sample_annotation_candidates(
        pairs, linkage, pairs_per_stratum=1, seed=42
    )
    pair_a = semantic_audit.stable_pair_id("private original", "private variant A")
    pair_d = semantic_audit.stable_pair_id("private original", "private variant D")
    linkage["selected_for_review"] = linkage["pair_id"].isin([pair_a, pair_d])
    annotations = pd.DataFrame([*_reviews(pair_a), *_reviews(pair_d, ("no", "no", "yes"))])
    consensus = semantic_audit.resolve_annotations(
        annotations, [pair_a, pair_d], ["reviewer-a", "reviewer-b"]
    )
    summary = semantic_audit.review_summary(pairs, linkage, consensus)

    # Union weights are 1/(2/3)=1.5 and 1/(1/2)=2, so validity is 1.5/3.5.
    assert summary["design_weighted_validity_estimate"] == pytest.approx(3 / 7)


def test_two_complete_disagreements_are_in_progress_instead_of_pending() -> None:
    pairs, candidates, linkage = _sample_all()
    annotations = pd.DataFrame(_reviews(candidates.iloc[0]["pair_id"]))
    annotations.loc[1, "semantic_preserved"] = "no"
    consensus = semantic_audit.resolve_annotations(
        annotations, candidates["pair_id"].tolist(), ["reviewer-a", "reviewer-b"]
    )
    summary = semantic_audit.review_summary(pairs, linkage, consensus)

    assert summary["status"] == "review_in_progress"
    assert summary["two_reviewer_complete_pairs"] == 1
    assert summary["resolved_pairs"] == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("original", None, "strings"),
        ("perturbed", 12, "strings"),
        ("success", "False", "boolean"),
        ("result_type", "", "nonempty"),
    ],
)
def test_malformed_attack_records_fail_without_exposing_private_text(
    field: str, value: Any, message: str
) -> None:
    record = _record("private original", "private variant")
    record[field] = value
    with pytest.raises(ValueError, match=message) as error:
        semantic_audit.build_pair_population(
            {"model-a__recipe-a": [record]}, {"private original": "source-a"}
        )
    assert "private original" not in str(error.value)


def test_malformed_cell_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="surrogate__recipe"):
        semantic_audit.build_pair_population({"invalid": []}, {})


def test_unrecognized_non_success_status_preserves_other_attempt_count() -> None:
    _, linkage = semantic_audit.build_pair_population(
        {
            "model-a__recipe-a": [
                _record("private original", "private variant", result_type="MaximizedAttackResult")
            ]
        },
        {"private original": "source-a"},
    )
    assert linkage["source_other_count"].sum() == 1


def test_annotation_truthiness_does_not_replace_explicit_label_vocabulary() -> None:
    annotations = pd.DataFrame(_reviews("pair-a"))
    annotations.loc[0, "intent_preserved"] = "probably"
    with pytest.raises(ValueError, match="yes, no, uncertain"):
        semantic_audit.resolve_annotations(annotations, ["pair-a"], ["reviewer-a", "reviewer-b"])


def test_missing_independence_attestation_cannot_resolve_a_pair() -> None:
    annotations = pd.DataFrame(_reviews("pair-a"))
    annotations.loc[0, "independent_review"] = ""
    consensus = semantic_audit.resolve_annotations(
        annotations, ["pair-a"], ["reviewer-a", "reviewer-b"]
    )
    assert consensus.iloc[0]["status"] == "unresolved"
    assert consensus.iloc[0]["complete_review_count"] == 1


def test_duplicate_selection_and_incomplete_consensus_are_rejected() -> None:
    with pytest.raises(ValueError, match="deduplicated"):
        semantic_audit.resolve_annotations(pd.DataFrame(), ["pair-a", "pair-a"], [])
    pairs, candidates, linkage = _sample_all()
    consensus = semantic_audit.resolve_annotations(candidates, candidates["pair_id"].tolist(), [])
    with pytest.raises(ValueError, match="exactly"):
        semantic_audit.review_summary(pairs, linkage, consensus.iloc[:1])


def test_public_summaries_contain_only_aggregates_and_no_pair_ids_or_raw_text() -> None:
    pairs, candidates, linkage = _sample_all()
    consensus = semantic_audit.resolve_annotations(candidates, candidates["pair_id"].tolist(), [])
    payload = json.dumps(
        {
            "fidelity": semantic_audit.fidelity_summary(pairs, linkage),
            "reviews": semantic_audit.review_summary(pairs, linkage, consensus),
        }
    )

    assert "private original" not in payload
    assert "private variant" not in payload
    assert all(pair_id not in payload for pair_id in pairs["pair_id"])
    assert "reviewer-a" not in payload


def test_empty_population_has_no_defined_validity_rate() -> None:
    pairs, linkage = semantic_audit.build_pair_population({}, {})
    candidates, linkage = semantic_audit.sample_annotation_candidates(
        pairs, linkage, pairs_per_stratum=1, seed=42
    )
    consensus = semantic_audit.resolve_annotations(candidates, [], [])
    summary = semantic_audit.review_summary(pairs, linkage, consensus)

    assert summary["population_pairs"] == 0
    assert summary["finite_population_pair_validity_bounds"] == {"lower": None, "upper": None}
    assert semantic_audit.fidelity_summary(pairs, linkage)["population"]["attempts"] == 0


def test_sampling_rejects_zero_draws_instead_of_creating_unreachable_units() -> None:
    pairs, linkage = _population()
    with pytest.raises(ValueError, match="positive"):
        semantic_audit.sample_annotation_candidates(pairs, linkage, pairs_per_stratum=0, seed=42)
