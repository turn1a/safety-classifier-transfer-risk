import json

import pandas as pd
import pytest

from transfer_risk.lib.retrospective_comparison import compare_saved_candidate_policies


def _evidence() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    frames = {}
    reports = []
    for target, models in (("reference", ["a", "b", "c"]), ("other", ["a", "b"])):
        rows = []
        for index, model in enumerate(models):
            rows.append(
                {
                    "target": target,
                    "surrogate": model,
                    "recipe": "recipe",
                    "attempted": 10,
                    "source_successful": 5,
                    "source_failed": 4,
                    "source_skipped": 1,
                    "source_other": 0,
                    "target_original_positive": 8,
                    "target_original_benign": 2,
                    "target_original_positive_successful": 4,
                    "verified_flips_source_successful": index + 1,
                    "verified_flips_source_failed": 1,
                    "verified_flips_source_skipped": 0,
                    "verified_flips_source_other": 0,
                    "verified_flips_all_candidates": index + 2,
                }
            )
        frames[target] = pd.DataFrame(rows)
        reports.append({"target": target, "included_surrogates": models})
    similarity = pd.DataFrame({"surrogate": ["a", "b", "c"], "mean_cka": [0.1, 0.2, 0.3]})
    return frames, similarity, {"reports": reports, "reference_check": {"counts_agree": True}}


def test_common_pool_uses_same_sources_and_counts_failed_candidates() -> None:
    frames, similarity, context = _evidence()
    report = compare_saved_candidate_policies(
        frames,
        similarity,
        context,
        attempts_per_cell=10,
        recipes=["recipe"],
        reference_target="reference",
    )
    assert report["common_surrogates"] == ["a", "b"]
    assert report["common_pool"][0]["attempted"] == 20
    assert report["common_pool"][0]["source_gated_flips"] == 3
    assert report["common_pool"][0]["all_candidate_flips"] == 5
    assert report["common_pool"][0]["original_target_positive"] == 8
    assert report["reference_pool"]["association"]["rho"] == pytest.approx(1.0)
    assert report["independent_selection_validation"] is False
    assert "pair_id" not in json.dumps(report)


@pytest.mark.parametrize("change", ["duplicate", "missing_model", "budget", "counts", "target"])
def test_common_pool_rejects_incomplete_or_inconsistent_aggregates(change: str) -> None:
    frames, similarity, context = _evidence()
    frame = frames["reference"]
    if change == "duplicate":
        frames["reference"] = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif change == "missing_model":
        frames["reference"] = frame.iloc[:2]
    elif change == "budget":
        frame.loc[0, "attempted"] = 9
    elif change == "counts":
        frame.loc[0, "verified_flips_all_candidates"] = 9
    else:
        frame.loc[0, "target"] = "unexpected"
    with pytest.raises(ValueError):
        compare_saved_candidate_policies(
            frames,
            similarity,
            context,
            attempts_per_cell=10,
            recipes=["recipe"],
            reference_target="reference",
        )
