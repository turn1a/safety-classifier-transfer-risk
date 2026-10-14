"""Finite-population valid-flip bounds and probability-weighted review reduction."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from transfer_risk.lib.semantic_yield import summarize_valid_flip_yield


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build hand-counted repeated trials with an excluded source and shared pairs."""
    predictions = pd.DataFrame(
        [
            {
                "pair_id": f"private-pair-{index}",
                "original_id": f"original-{index}",
                "original_pred": 0 if index == 4 else 1,
                "perturbed_pred": 0,
            }
            for index in range(1, 6)
        ]
    )
    specifications = [
        (1, "model-a", "r1", 2, 1, True, 0.5),
        (1, "model-b", "r1", 1, 1, True, 0.5),
        (2, "model-a", "r2", 2, 1, True, 0.25),
        (3, "model-b", "r2", 4, 0, False, 0.5),
        (4, "model-a", "r3", 1, 1, False, 0.5),
        (5, "bilstm-attention", "r1", 5, 5, False, 0.5),
    ]
    linkage = pd.DataFrame(
        [
            {
                "pair_id": f"private-pair-{pair}",
                "original_id": f"original-{pair}",
                "surrogate": model,
                "recipe": recipe,
                "source": "corpus-a",
                "attempt_count": count,
                "source_successful_count": successes,
                "source_failed_count": count - successes,
                "source_skipped_count": 0,
                "source_other_count": 0,
                "selected_for_review": selected,
                "inclusion_probability": probability,
                "design_weight": 1 / probability,
            }
            for pair, model, recipe, count, successes, selected, probability in specifications
        ]
    )
    consensus = pd.DataFrame(
        {"pair_id": ["private-pair-1", "private-pair-2"], "status": ["valid", "invalid"]}
    )
    return consensus, predictions, linkage


def _summary(
    consensus: pd.DataFrame, predictions: pd.DataFrame, linkage: pd.DataFrame
) -> dict[str, Any]:
    """Apply the reference target's explicit source exclusion."""
    return summarize_valid_flip_yield(
        consensus,
        predictions,
        linkage,
        target_id="reference",
        excluded_surrogates=["bilstm-attention"],
    )


def test_policy_bounds_keep_all_trials_in_the_yield_denominator_and_preserve_multiplicity() -> None:
    summary = _summary(*_inputs())
    gated = summary["policies"]["source_success_gated"]
    all_returned = summary["policies"]["all_returned_candidates"]

    assert summary["attempted_trials"] == 10
    assert gated["classifier_flip_trials"] == 3
    assert gated["confirmed_valid_flip_trials"] == 2
    assert gated["confirmed_invalid_flip_trials"] == 1
    assert gated["unresolved_flip_trials"] == 0
    assert gated["valid_flip_count_bounds"] == {"lower": 2, "upper": 2}
    assert gated["valid_flip_yield_bounds"] == {"lower": 0.2, "upper": 0.2}
    assert all_returned["classifier_flip_trials"] == 9
    assert all_returned["confirmed_valid_flip_trials"] == 3
    assert all_returned["confirmed_invalid_flip_trials"] == 2
    assert all_returned["unresolved_flip_trials"] == 4
    assert all_returned["valid_flip_count_bounds"] == {"lower": 3, "upper": 7}
    assert all_returned["valid_flip_yield_bounds"] == {"lower": 0.3, "upper": 0.7}


def test_flip_coverage_is_separate_from_validity_bounds_and_yield_estimates_are_withheld() -> None:
    summary = _summary(*_inputs())
    result = summary["policies"]["all_returned_candidates"]

    assert result["classifier_flip_pairs"] == 3
    assert result["sampled_classifier_flip_pairs"] == 2
    assert result["resolved_classifier_flip_pairs"] == 2
    assert result["sampled_classifier_flip_trials"] == 5
    assert result["resolved_classifier_flip_trials"] == 5
    assert result["resolved_classifier_flip_fraction"] == pytest.approx(5 / 9)
    assert result["estimate"] is None
    assert summary["policies"]["source_success_gated"]["estimate"] is None
    assert summary["bounds_are_confidence_intervals"] is False
    assert summary["cell_rates_estimated"] is False
    assert summary["original_union_estimated"] is False


def test_pending_or_partial_sample_never_supplies_a_point_estimate() -> None:
    consensus, predictions, linkage = _inputs()
    consensus.loc[0, "status"] = "unresolved"
    summary = _summary(consensus, predictions, linkage)
    assert summary["selected_judgments_resolved"] is False
    for policy in summary["policies"].values():
        assert policy["estimate"] is None
    all_returned = summary["policies"]["all_returned_candidates"]
    assert all_returned["valid_flip_count_bounds"] == {"lower": 0, "upper": 7}
    assert all_returned["unresolved_flip_trials"] == 7


def test_reviewed_pairs_do_not_stand_in_for_any_unsampled_pairs() -> None:
    consensus, predictions, linkage = _inputs()
    linkage.loc[linkage["pair_id"] == "private-pair-1", "inclusion_probability"] = 1.0
    linkage["design_weight"] = 1 / linkage["inclusion_probability"]
    linkage.loc[linkage["pair_id"] == "private-pair-2", "selected_for_review"] = False
    consensus = consensus.iloc[:1]
    summary = _summary(consensus, predictions, linkage)
    assert summary["selected_judgments_resolved"] is True
    assert summary["policies"]["all_returned_candidates"]["estimate"] is None
    assert summary["policies"]["all_returned_candidates"]["valid_flip_count_bounds"] == {
        "lower": 3,
        "upper": 9,
    }


def test_full_observed_flip_validity_has_exact_bounds_even_with_unequal_design_weights() -> None:
    consensus, predictions, linkage = _inputs()
    linkage.loc[linkage["pair_id"] == "private-pair-3", "selected_for_review"] = True
    consensus = pd.concat(
        [consensus, pd.DataFrame({"pair_id": ["private-pair-3"], "status": ["valid"]})],
        ignore_index=True,
    )
    result = _summary(consensus, predictions, linkage)["policies"]["all_returned_candidates"]
    assert result["valid_flip_count_bounds"] == {"lower": 7, "upper": 7}
    assert result["estimate"] is None


def test_consensus_must_cover_the_exact_global_sample_and_design_must_agree_per_pair() -> None:
    consensus, predictions, linkage = _inputs()
    with pytest.raises(ValueError, match="selected"):
        _summary(consensus.iloc[:1], predictions, linkage)
    linkage.loc[1, "selected_for_review"] = False
    with pytest.raises(ValueError, match="pair"):
        _summary(consensus, predictions, linkage)


def test_invalid_predictions_and_status_counts_are_rejected_by_retrospective_validation() -> None:
    consensus, predictions, linkage = _inputs()
    with pytest.raises(ValueError, match="missing"):
        _summary(consensus, predictions.iloc[1:], linkage)
    linkage.loc[0, "source_failed_count"] = 100
    with pytest.raises(ValueError, match="sum"):
        _summary(consensus, predictions, linkage)


def test_summary_excludes_pair_ids_original_ids_raw_text_and_cell_estimates() -> None:
    summary = _summary(*_inputs())
    serialized = json.dumps(summary)
    assert "private-pair" not in serialized
    assert all(f"original-{index}" not in serialized for index in range(1, 6))
    assert "corpus-a" not in serialized
    assert summary["cka_validity_association"] is None


def test_no_classifier_flips_has_zero_bounds_and_undefined_flip_review_fraction() -> None:
    consensus, predictions, linkage = _inputs()
    predictions["perturbed_pred"] = predictions["original_pred"]
    result = _summary(consensus, predictions, linkage)["policies"]["all_returned_candidates"]
    assert result["valid_flip_count_bounds"] == {"lower": 0, "upper": 0}
    assert result["valid_flip_yield_bounds"] == {"lower": 0.0, "upper": 0.0}
    assert result["resolved_classifier_flip_fraction"] is None


def test_invalid_consensus_states_and_nonboolean_selection_are_rejected() -> None:
    consensus, predictions, linkage = _inputs()
    consensus.loc[0, "status"] = "probably valid"
    with pytest.raises(ValueError, match="status"):
        _summary(consensus, predictions, linkage)
    consensus.loc[0, "status"] = "valid"
    linkage = linkage.astype({"selected_for_review": "object"})
    linkage.loc[0, "selected_for_review"] = "False"
    with pytest.raises(ValueError, match="boolean"):
        _summary(consensus, predictions, linkage)
