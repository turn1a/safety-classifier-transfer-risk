"""Behavioral checks for all-candidate retrospective transfer accounting."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from transfer_risk.lib.retrospective_audit import build_retrospective_cells


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.DataFrame(
        [
            {"pair_id": "a", "original_id": "o1", "original_pred": 1, "perturbed_pred": 0},
            {"pair_id": "b", "original_id": "o2", "original_pred": 1, "perturbed_pred": 0},
            {"pair_id": "c", "original_id": "o3", "original_pred": 0, "perturbed_pred": 0},
            {"pair_id": "d", "original_id": "o4", "original_pred": 1, "perturbed_pred": 1},
        ]
    )
    linkage = pd.DataFrame(
        [
            {
                "pair_id": pair,
                "original_id": original,
                "source": "source-a",
                "surrogate": "s1",
                "recipe": "r1",
                "attempt_count": 1,
                "source_successful_count": successful,
                "source_failed_count": failed,
                "source_skipped_count": skipped,
                "source_other_count": 0,
            }
            for pair, original, successful, failed, skipped in [
                ("a", "o1", 1, 0, 0),
                ("b", "o2", 0, 1, 0),
                ("c", "o3", 1, 0, 0),
                ("d", "o4", 0, 0, 1),
            ]
        ]
    )
    return predictions, linkage


def test_counts_failed_source_candidates_without_calling_them_source_successes() -> None:
    predictions, linkage = _inputs()
    cells, sources, context = build_retrospective_cells(predictions, linkage, target_id="t")
    row = cells.iloc[0]
    assert row["attempted"] == 4
    assert row["source_successful"] == 2
    assert row["verified_flips_source_successful"] == 1
    assert row["verified_flips_source_failed"] == 1
    assert row["verified_flips_all_candidates"] == 2
    assert row["source_gated_yield"] == 0.25
    assert row["all_candidate_yield"] == 0.5
    assert row["source_conditional_flip_rate"] == 1.0
    assert row["all_candidate_conditional_flip_rate"] == pytest.approx(2 / 3)
    assert sources.iloc[0]["originally_benign_candidates_benign"] == 1
    assert context["unique_originals"] == 4
    assert context["unique_original_positive"] == 3
    assert context["semantic_validity"] == "not_assessed"
    assert context["application_effectiveness"] == "not_assessed"
    json.dumps(context, allow_nan=False)


def test_excludes_target_self_source_and_named_defective_source() -> None:
    predictions, linkage = _inputs()
    linkage.loc[0, "surrogate"] = "t-self"
    linkage.loc[1, "surrogate"] = "bilstm-attention"
    cells, _, context = build_retrospective_cells(
        predictions,
        linkage,
        target_id="t",
        excluded_surrogates=("t-self", "bilstm-attention"),
    )
    assert set(cells["surrogate"]) == {"s1"}
    assert cells["attempted"].sum() == 2
    assert context["aggregate_counts"]["verified_flips_all_candidates"] == 0
    assert context["excluded_surrogates"] == ["bilstm-attention", "t-self"]


def test_preserves_record_multiplicity_and_never_counts_already_benign_as_flip() -> None:
    predictions, linkage = _inputs()
    linkage.loc[2, ["attempt_count", "source_successful_count"]] = 7
    cells, _, context = build_retrospective_cells(predictions, linkage, target_id="t")
    assert cells.iloc[0]["attempted"] == 10
    assert cells.iloc[0]["verified_flips_all_candidates"] == 2
    assert cells.iloc[0]["originally_benign_candidates_benign"] == 7
    assert context["unique_originals"] == 4


def test_zero_positive_denominator_produces_json_null() -> None:
    predictions, linkage = _inputs()
    predictions.loc[:, "original_pred"] = 0
    predictions.loc[:, "perturbed_pred"] = 0
    cells, _, context = build_retrospective_cells(predictions, linkage, target_id="t")
    assert pd.isna(cells.iloc[0]["source_conditional_flip_rate"])
    assert context["source_conditional_flip_rate"] is None
    assert context["all_candidate_conditional_flip_rate"] is None
    assert context["all_candidate_yield"] == 0.0


@pytest.mark.parametrize("problem", ["negative", "fraction", "status_sum", "missing_prediction"])
def test_rejects_unaccounted_trials(problem: str) -> None:
    predictions, linkage = _inputs()
    if problem == "negative":
        linkage.loc[0, "attempt_count"] = -1
    elif problem == "fraction":
        linkage = linkage.astype({"attempt_count": float})
        linkage.loc[0, "attempt_count"] = 1.5
    elif problem == "status_sum":
        linkage.loc[0, "source_failed_count"] = 1
    else:
        predictions = predictions.iloc[1:]
    with pytest.raises(ValueError):
        build_retrospective_cells(predictions, linkage, target_id="t")


def test_rejects_inconsistent_original_baselines_and_skipped_flips() -> None:
    predictions, linkage = _inputs()
    predictions.loc[3, "perturbed_pred"] = 0
    with pytest.raises(ValueError, match="skipped"):
        build_retrospective_cells(predictions, linkage, target_id="t")
    predictions, linkage = _inputs()
    predictions.loc[2, "original_id"] = "o1"
    linkage.loc[2, "original_id"] = "o1"
    with pytest.raises(ValueError, match="baseline"):
        build_retrospective_cells(predictions, linkage, target_id="t")


def test_public_tables_exclude_pair_ids_and_text() -> None:
    predictions, linkage = _inputs()
    cells, sources, context = build_retrospective_cells(predictions, linkage, target_id="t")
    private_names = {"pair_id", "original_id", "original", "perturbed", "notes"}
    assert not private_names.intersection(cells.columns)
    assert not private_names.intersection(sources.columns)
    assert not private_names.intersection(context)
