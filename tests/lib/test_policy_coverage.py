import pandas as pd
import pytest

from transfer_risk.lib.policy_coverage import summarize_original_policy_coverage


def _records() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions, linkage = [], []
    for source in ("a", "b", "c"):
        for recipe in ("first", "second"):
            for original in ("x", "y", "z"):
                pair = f"{source}-{recipe}-{original}"
                failed = source == "a" and original == "y"
                flip = (source in {"a", "b"} and original == "x") or (
                    source in {"a", "c"} and original == "y"
                )
                predictions.append(
                    {
                        "pair_id": pair,
                        "original_id": original,
                        "original_pred": int(original != "z"),
                        "perturbed_pred": int(original != "z" and not flip),
                    }
                )
                linkage.append(
                    {
                        "pair_id": pair,
                        "original_id": original,
                        "source": "dataset",
                        "surrogate": source,
                        "recipe": recipe,
                        "attempt_count": 1,
                        "source_successful_count": int(not failed),
                        "source_failed_count": int(failed),
                        "source_skipped_count": 0,
                        "source_other_count": 0,
                    }
                )
    return (
        pd.DataFrame(predictions),
        pd.DataFrame(linkage),
        pd.DataFrame({"surrogate": ["a", "b", "c"], "mean_cka": [0.9, 0.8, 0.7]}),
    )


def test_selection_coverage_counts_each_original_once_across_sources_and_recipes() -> None:
    predictions, linkage, similarity = _records()
    report = summarize_original_policy_coverage(
        predictions,
        linkage,
        similarity,
        attempts_per_cell=3,
        recipes=["first", "second"],
        excluded_surrogates=(),
        selection_size=2,
    )
    assert report["originals"] == 3
    assert report["original_target_positive"] == 2
    assert report["selected_surrogates"] == ["a", "b"]
    assert report["selected"]["source_gated_originals"] == 1
    assert report["selected"]["all_candidate_originals"] == 2
    assert report["comparisons"]["source_gated"]["n_strictly_higher"] == 2
    assert report["comparisons"]["all_candidate"]["n_strictly_higher"] == 0
    assert report["attempted_trials_per_subset"] == 12
    assert "original_id" not in str(report)


def test_selection_coverage_rejects_an_entire_missing_model() -> None:
    predictions, linkage, similarity = _records()
    linkage = linkage[linkage["surrogate"] != "c"]
    with pytest.raises(ValueError, match="grid"):
        summarize_original_policy_coverage(
            predictions,
            linkage,
            similarity,
            attempts_per_cell=3,
            recipes=["first", "second"],
            excluded_surrogates=(),
            selection_size=2,
        )


def test_selection_coverage_rejects_repeated_original_trials() -> None:
    predictions, linkage, similarity = _records()
    linkage.loc[0, "attempt_count"] = 2
    linkage.loc[0, "source_successful_count"] = 2
    with pytest.raises(ValueError, match=r"grid|once"):
        summarize_original_policy_coverage(
            predictions,
            linkage,
            similarity,
            attempts_per_cell=3,
            recipes=["first", "second"],
            excluded_surrogates=(),
            selection_size=2,
        )
