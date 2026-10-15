"""Trial-sampled human-validity census invariants on synthetic saved data."""

from __future__ import annotations

import itertools
import json
from typing import Any

import pandas as pd
import pytest
from scipy.stats import hypergeom

from transfer_risk.lib.semantic_audit import stable_pair_id
from transfer_risk.lib.semantic_flip_audit import (
    build_flip_cohort,
    hypergeometric_count_interval,
    summarize_flip_review,
    validate_flip_cohort,
)


def inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create exact pairs with repeated trials, source failures and a no-flip pair."""
    pairs = pd.DataFrame(
        [
            {
                "pair_id": stable_pair_id(f"original {i}", f"modified {i}"),
                "original_id": f"o{i}",
                "original": f"original {i}",
                "perturbed": f"modified {i}",
            }
            for i in range(4)
        ]
    )
    linkage = pd.DataFrame(
        [
            {
                "pair_id": pairs.iloc[i].pair_id,
                "original_id": f"o{i}",
                "source": "corpus",
                "surrogate": model,
                "recipe": "r",
                "attempt_count": count,
                "source_successful_count": success,
                "source_failed_count": count - success,
                "source_skipped_count": 0,
                "source_other_count": 0,
            }
            for i, model, count, success in [
                (0, "a", 3, 1),
                (1, "a", 1, 1),
                (2, "b", 2, 0),
                (3, "b", 1, 0),
                (0, "bilstm-attention", 2, 2),
            ]
        ]
    )
    predictions = pairs[["pair_id", "original_id"]].copy()
    predictions["original_pred"] = 1
    predictions["perturbed_pred"] = [0, 0, 0, 1]
    return pairs, linkage, predictions


def cohort(sample_size: int = 3) -> dict[str, Any]:
    """Build the versioned cohort using only the synthetic tables."""

    return build_flip_cohort(*inputs(), seed=14, sample_size=sample_size)


def test_srs_uses_trial_occurrences_preserves_multiplicity_and_is_input_order_independent() -> None:

    first = cohort()
    reordered = build_flip_cohort(*(frame.iloc[::-1] for frame in inputs()), seed=14, sample_size=3)
    assert first == reordered
    assert first["attempted_trials"] == 7
    assert len(first["trials"]) == 6
    assert len(first["sample_occurrence_ids"]) == 3
    assert len(set(first["sample_occurrence_ids"])) == 3
    assert first["review_dataset"]["pair_count"] == 3
    sample = set(first["sample_occurrence_ids"])
    sampled_pairs = {t["pair_id"] for t in first["trials"] if t["occurrence_id"] in sample}
    queue = first["review_dataset"]["pairs"]
    assert {p["pair_id"] for p in queue[: len(sampled_pairs)]} == sampled_pairs
    assert first["review_dataset"]["stage_one_pair_count"] == len(sampled_pairs)
    assert all(t["surrogate"] != "bilstm-attention" for t in first["trials"])
    # Every subset of occurrence identities is eligible for the SRS draw.
    draws = [
        set(build_flip_cohort(*inputs(), seed=s, sample_size=3)["sample_occurrence_ids"])
        for s in range(250)
    ]
    assert len({tuple(sorted(draw)) for draw in draws}) == len(
        list(itertools.combinations(range(6), 3))
    )
    assert any(
        len({t["pair_id"] for t in first["trials"] if t["occurrence_id"] in draw}) < 3
        for draw in draws
    )


def test_exact_hypergeometric_count_intervals_have_at_least_nominal_exhaustive_coverage() -> None:

    for population in range(1, 13):
        for sample in range(1, population + 1):
            for total in range(population + 1):
                coverage = 0.0
                for observed in range(sample + 1):
                    lower, upper = hypergeometric_count_interval(population, sample, observed)
                    if lower <= total <= upper:
                        coverage += hypergeom.pmf(observed, population, total, sample)
                assert coverage >= 0.95 - 1e-12
    assert hypergeometric_count_interval(6, 6, 2) == (2, 2)
    for args in [(0, 1, 0), (5, 6, 0), (5, 2, 3), (5, 0, 0)]:
        with pytest.raises(ValueError):
            hypergeometric_count_interval(*args)


def test_missing_reviews_widen_interim_intervals_and_keep_census_denominators() -> None:

    saved = cohort()
    summary = summarize_flip_review(saved, pd.DataFrame(columns=["pair_id", "status"]))
    assert summary["review_status"] == "pending"
    assert summary["attempted_trials"] == 7
    for policy, flips in [("all_returned_candidates", 6), ("source_success_gated", 2)]:
        result = summary["policies"][policy]
        assert result["classifier_flip_trials"] == flips
        assert result["confirmed_valid_flip_trials"] == 0
        assert result["unresolved_flip_trials"] == flips
        assert result["valid_flip_yield_bounds"] == {"lower": 0, "upper": flips / 7}
        assert result["interim"]["population_count_interval"]["upper"] <= flips
        if result["interim"]["sample_unknown_trials"]:
            assert result["interim"]["point_estimate"] is None
        assert result["association"]["rho"] is None
    assert "pair_id" not in json.dumps(summary)
    assert not any(
        pair["original"] in json.dumps(summary) for pair in saved["review_dataset"]["pairs"]
    )


def test_census_counts_cells_original_union_and_null_outcome_are_exact() -> None:

    saved = cohort(6)
    statuses = pd.DataFrame(
        {"pair_id": [p["pair_id"] for p in saved["review_dataset"]["pairs"]], "status": "valid"}
    )
    summary = summarize_flip_review(
        saved, statuses, pd.DataFrame({"surrogate": ["a", "b"], "mean_cka": [0.2, 0.7]})
    )
    assert summary["review_status"] == "census_resolved"
    result = summary["policies"]["all_returned_candidates"]
    assert result["confirmed_valid_flip_trials"] == 6
    assert result["interim"]["population_count_interval"] == {"lower": 6, "upper": 6}
    assert result["original_union"]["valid_original_count_bounds"] == {"lower": 3, "upper": 3}
    assert result["original_union"]["originals"] == 4
    assert result["association"]["n"] == 2
    assert sum(cell["attempted_trials"] for cell in summary["cells"]) == 14  # two policies
    statuses["status"] = "invalid"
    null = summarize_flip_review(
        saved, statuses, pd.DataFrame({"surrogate": ["a", "b"], "mean_cka": [0.2, 0.7]})
    )
    assert null["policies"]["all_returned_candidates"]["association"]["rho"] is None
    assert null["policies"]["all_returned_candidates"]["interim"]["point_estimate"] == 0


def test_consensus_unknown_pairs_duplicates_and_changed_cohort_are_rejected() -> None:

    saved = cohort()
    for frame in [
        pd.DataFrame({"pair_id": ["outside"], "status": ["valid"]}),
        pd.DataFrame({"pair_id": [saved["trials"][0]["pair_id"]] * 2, "status": ["valid"] * 2}),
    ]:
        with pytest.raises(ValueError):
            summarize_flip_review(saved, frame)
    saved["trials"][0]["source_success"] = not saved["trials"][0]["source_success"]
    with pytest.raises(ValueError, match="fingerprint"):
        validate_flip_cohort(saved)


def test_unresolved_sample_count_interval_is_envelope_and_point_is_withheld() -> None:

    saved = cohort()
    statuses = pd.DataFrame(
        {"pair_id": [saved["review_dataset"]["pairs"][0]["pair_id"]], "status": ["valid"]}
    )
    result = summarize_flip_review(saved, statuses)["policies"]["all_returned_candidates"][
        "interim"
    ]
    assert result["sample_unknown_trials"] > 0
    assert result["point_estimate"] is None
    low = hypergeometric_count_interval(6, 3, result["sample_confirmed_valid_trials"])[0]
    high = hypergeometric_count_interval(
        6, 3, result["sample_confirmed_valid_trials"] + result["sample_unknown_trials"]
    )[1]
    assert result["population_count_interval"] == {"lower": low, "upper": high}


def test_review_completion_and_stage_two_preserve_resolution_and_sample_size() -> None:
    saved = cohort()
    statuses = pd.DataFrame(
        {
            "pair_id": [pair["pair_id"] for pair in saved["review_dataset"]["pairs"]],
            "status": "unresolved",
            "reviewer_count": 2,
            "complete_review_count": 2,
        }
    )
    summary = summarize_flip_review(saved, statuses)
    assert summary["two_reviewer_complete_pairs"] == 3
    assert summary["resolved_pairs"] == 0
    assert all(value["interim"]["sample_trials"] == 3 for value in summary["policies"].values())
    assert (
        sum(
            row["attempted_trials"]
            for row in summary["surrogates"]
            if row["policy"] == "all_returned_candidates"
        )
        == 7
    )


def test_zero_flip_population_requires_no_reviews_and_retains_all_attempts() -> None:
    pairs, linkage, predictions = inputs()
    predictions["perturbed_pred"] = 1
    saved = build_flip_cohort(pairs, linkage, predictions, seed=14, sample_size=3)
    summary = summarize_flip_review(saved, pd.DataFrame(columns=["pair_id", "status"]))
    assert summary["distinct_pair_workload"] == 0
    assert summary["attempted_trials"] == 7
    assert summary["policies"]["all_returned_candidates"]["interim"]["point_estimate"] == 0


def test_failed_source_sample_trials_are_identified_gated_zeros_without_human_labels() -> None:
    saved = cohort()
    result = summarize_flip_review(saved, pd.DataFrame(columns=["pair_id", "status"]))
    gated = result["policies"]["source_success_gated"]["interim"]
    assert gated["sample_unknown_trials"] == 0
    assert gated["point_estimate"] == 0.0
    assert result["policies"]["all_returned_candidates"]["interim"]["point_estimate"] is None
