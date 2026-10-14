import pandas as pd
import pytest

from transfer_risk.lib.source_baseline_check import verify_source_baseline_counts


def _data() -> tuple[pd.DataFrame, dict]:
    rows, reports = [], []
    for model in ("a", "b"):
        by_source = []
        for source, positive, missed in (("x", 3, 1), ("y", 2, 0)):
            by_source.append(
                {
                    "source": source,
                    "canonical_positive_count": positive,
                    "canonical_positive_predicted_negative_count": missed,
                }
            )
            for recipe in ("first", "second"):
                rows.append(
                    {
                        "surrogate": model,
                        "source": source,
                        "recipe": recipe,
                        "attempt_count": positive,
                        "source_skipped_count": missed,
                    }
                )
        reports.append({"target_id": model, "by_source": by_source})
    return pd.DataFrame(rows), {"reports": reports}


def test_source_eligibility_check_requires_per_dataset_and_recipe_count_agreement() -> None:
    linkage, baseline = _data()
    result = verify_source_baseline_counts(
        linkage, baseline, recipes=["first", "second"], attempts_per_cell=5
    )
    assert result["counts_agree"] is True
    assert result["cells_checked"] == 8
    assert result["models_checked"] == ["a", "b"]
    assert "pair_id" not in str(result)


def test_pooled_agreement_cannot_hide_source_baseline_mismatch() -> None:
    linkage, baseline = _data()
    linkage.loc[0, "source_skipped_count"] = 0
    linkage.loc[2, "source_skipped_count"] = 1
    with pytest.raises(ValueError, match="baseline"):
        verify_source_baseline_counts(
            linkage, baseline, recipes=["first", "second"], attempts_per_cell=5
        )


def test_source_eligibility_check_rejects_missing_whole_model() -> None:
    linkage, baseline = _data()
    with pytest.raises(ValueError, match="model"):
        verify_source_baseline_counts(
            linkage[linkage.surrogate != "b"],
            baseline,
            recipes=["first", "second"],
            attempts_per_cell=5,
        )


def test_source_baseline_check_rejects_fractional_misses_in_public_summary() -> None:
    linkage, baseline = _data()
    baseline["reports"][0]["by_source"][0]["canonical_positive_predicted_negative_count"] = 1.5
    with pytest.raises(ValueError, match="integer"):
        verify_source_baseline_counts(
            linkage, baseline, recipes=["first", "second"], attempts_per_cell=5
        )
