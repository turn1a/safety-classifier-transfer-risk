"""Tests for transfer_risk.lib.target_audit: post-hoc target-outcome audit helpers."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from transfer_risk.lib import target_audit


def _annotated_record(
    *,
    surrogate: str = "s1",
    recipe: str = "r1",
    source: str = "deepset/prompt-injections",
    original_pred: int = 1,
    perturbed_pred: int = 0,
) -> dict[str, int | str]:
    """Build one annotated audit record without raw text."""
    return {
        "surrogate": surrogate,
        "recipe": recipe,
        "source": source,
        "original_pred": original_pred,
        "perturbed_pred": perturbed_pred,
    }


def test_truncate_prompt_matches_attack_time_cut() -> None:
    text = "x" * 100
    assert target_audit.truncate_prompt(text, 40) == "x" * 40


def test_build_eval_original_source_map_is_total_for_eval_set() -> None:
    test_df = pd.DataFrame(
        [
            {"text": "alpha prompt", "label": 1, "source": "deepset/prompt-injections"},
            {"text": "beta prompt", "label": 1, "source": "jackhhao/jailbreak-classification"},
            {"text": "benign", "label": 0, "source": "deepset/prompt-injections"},
        ]
    )
    mapping = target_audit.build_eval_original_source_map(
        test_df, eval_set_size=2, max_prompt_chars=3200
    )
    assert mapping["alpha prompt"] == "deepset/prompt-injections"
    assert mapping["beta prompt"] == "jackhhao/jailbreak-classification"


def test_build_eval_original_source_map_rejects_ambiguous_truncation() -> None:
    test_df = pd.DataFrame(
        [
            {"text": "sameprefix-A", "label": 1, "source": "deepset/prompt-injections"},
            {"text": "sameprefix-B", "label": 1, "source": "jackhhao/jailbreak-classification"},
        ]
    )
    with pytest.raises(ValueError, match="ambiguous"):
        target_audit.build_eval_original_source_map(test_df, eval_set_size=2, max_prompt_chars=10)


def test_assign_sources_requires_total_mapping() -> None:
    records = [{"original": "missing", "success": True}]
    with pytest.raises(ValueError, match="unmapped"):
        target_audit.assign_sources_to_records(records, {}, max_prompt_chars=3200)


def test_dedupe_texts_preserves_first_seen_order() -> None:
    assert target_audit.dedupe_texts(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_aggregate_audit_counts_and_true_flip_rate() -> None:
    records = [
        _annotated_record(original_pred=1, perturbed_pred=0),
        _annotated_record(original_pred=1, perturbed_pred=1),
        _annotated_record(original_pred=0, perturbed_pred=0),
    ]
    stats = target_audit.aggregate_audit_counts(records, benign_label=0, injection_label=1)
    assert stats["source_successful"] == 3
    assert stats["target_original_benign"] == 1
    assert stats["target_original_injection"] == 2
    assert stats["target_perturbed_benign"] == 2
    assert stats["true_target_flips"] == 1
    assert stats["conditional_target_benign_rate"] == pytest.approx(2 / 3)
    assert stats["true_target_flip_rate"] == pytest.approx(0.5)


def test_aggregate_audit_counts_uses_null_for_every_undefined_rate() -> None:
    """Undefined rates remain null for empty and true-flip-zero-denominator groups."""
    no_original_injections = target_audit.aggregate_audit_counts(
        [_annotated_record(original_pred=0, perturbed_pred=0)],
        benign_label=0,
        injection_label=1,
    )
    empty = target_audit.aggregate_audit_counts([], benign_label=0, injection_label=1)

    assert no_original_injections["conditional_target_benign_rate"] == pytest.approx(1.0)
    assert no_original_injections["target_original_injection"] == 0
    assert no_original_injections["true_target_flip_rate"] is None
    assert empty["source_successful"] == 0
    assert empty["conditional_target_benign_rate"] is None
    assert empty["true_target_flip_rate"] is None


def test_aggregate_by_cell_and_source_frames_have_no_text_columns() -> None:
    records = [
        _annotated_record(source="deepset/prompt-injections"),
        _annotated_record(
            surrogate="s1",
            recipe="r2",
            source="jackhhao/jailbreak-classification",
            original_pred=1,
            perturbed_pred=1,
        ),
    ]
    cells = target_audit.aggregate_by_cell(records, benign_label=0, injection_label=1)
    sources = target_audit.aggregate_by_source(records, benign_label=0, injection_label=1)
    for frame in (cells, sources):
        assert "original" not in frame.columns
        assert "perturbed" not in frame.columns
        assert "text" not in frame.columns


def test_validate_source_aggregate_rollups_accepts_exact_cell_reconciliation() -> None:
    """Accept source rows whose count fields sum exactly to each full cell."""
    raw_cells = pd.DataFrame(
        [
            {
                "surrogate": "s1",
                "recipe": "r1",
                "source_successful": 3,
                "target_original_benign": 1,
                "target_original_injection": 2,
                "target_perturbed_benign": 2,
                "true_target_flips": 1,
            }
        ]
    )
    raw_sources = pd.DataFrame(
        [
            {
                "surrogate": "s1",
                "recipe": "r1",
                "source": "source-a",
                "source_successful": 1,
                "target_original_benign": 0,
                "target_original_injection": 1,
                "target_perturbed_benign": 1,
                "true_target_flips": 1,
            },
            {
                "surrogate": "s1",
                "recipe": "r1",
                "source": "source-b",
                "source_successful": 2,
                "target_original_benign": 1,
                "target_original_injection": 1,
                "target_perturbed_benign": 1,
                "true_target_flips": 0,
            },
        ]
    )

    assert target_audit.validate_source_aggregate_rollups(raw_cells, raw_sources) is None


def test_validate_source_aggregate_rollups_rejects_mismatched_source_counts() -> None:
    """Reject a source rollup that cannot reconcile to its full cell."""
    raw_cells = pd.DataFrame(
        [
            {
                "surrogate": "s1",
                "recipe": "r1",
                "source_successful": 3,
                "target_original_benign": 1,
                "target_original_injection": 2,
                "target_perturbed_benign": 2,
                "true_target_flips": 1,
            }
        ]
    )
    raw_sources = pd.DataFrame(
        [
            {
                "surrogate": "s1",
                "recipe": "r1",
                "source": "source-a",
                "source_successful": 4,
                "target_original_benign": 1,
                "target_original_injection": 3,
                "target_perturbed_benign": 2,
                "true_target_flips": 1,
            }
        ]
    )

    with pytest.raises(ValueError, match="source aggregate rollups"):
        target_audit.validate_source_aggregate_rollups(raw_cells, raw_sources)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_successful", -1, "must contain non-negative integers"),
        ("source", None, "must define a source"),
    ],
)
def test_validate_source_aggregate_rollups_rejects_malformed_source_rows(
    field: str,
    value: int | None,
    message: str,
) -> None:
    """Reject invalid count scalars and missing source provenance before rollup."""
    raw_cells = pd.DataFrame(
        [
            {
                "surrogate": "s1",
                "recipe": "r1",
                "source_successful": 1,
                "target_original_benign": 0,
                "target_original_injection": 1,
                "target_perturbed_benign": 1,
                "true_target_flips": 1,
            }
        ]
    )
    raw_sources = raw_cells.assign(source="source-a")
    raw_sources.loc[0, field] = value

    with pytest.raises(ValueError, match=message):
        target_audit.validate_source_aggregate_rollups(raw_cells, raw_sources)


def test_finalize_audit_from_aggregates_rejects_unreconciled_source_counts() -> None:
    """Finalization must reject a source aggregate before rebuilding any audit grid."""
    raw_cells = pd.DataFrame(
        [
            {
                "surrogate": "s1",
                "recipe": "r1",
                "source_successful": 1,
                "target_original_benign": 0,
                "target_original_injection": 1,
                "target_perturbed_benign": 1,
                "true_target_flips": 1,
            }
        ]
    )
    raw_sources = raw_cells.assign(source="source-a", source_successful=2)
    raw_context = {
        "excluded_source": target_audit.KNOWN_TARGET_TRAINING_SOURCE,
        "unique_text_counts": {},
        "target_baseline_on_unique_originals": {},
    }

    with pytest.raises(ValueError, match="do not reconcile"):
        target_audit.finalize_audit_from_aggregates(
            raw_cells,
            raw_sources,
            raw_context,
            pd.DataFrame(),
            {},
            {},
            {},
            {},
            seed=17,
        )


def test_finalize_audit_from_aggregates_rejects_invalid_context_counts() -> None:
    """Finalization cannot consume negative counts from its non-text context."""
    raw_context = {
        "excluded_source": target_audit.KNOWN_TARGET_TRAINING_SOURCE,
        "unique_text_counts": {"unique_originals": -1},
        "target_baseline_on_unique_originals": {},
    }

    with pytest.raises(ValueError, match="has invalid count"):
        target_audit.finalize_audit_from_aggregates(
            pd.DataFrame(),
            pd.DataFrame(),
            raw_context,
            pd.DataFrame(),
            {},
            {},
            {},
            {},
            seed=17,
        )


def _complete_master_grid() -> pd.DataFrame:
    """Build an in-memory five-surrogate, two-recipe master grid."""
    similarities = {
        "s1": (0.95, 0.85),
        "s2": (0.75, 0.65),
        "s3": (0.25, 0.35),
        "s4": (0.05, 0.15),
        "s5": (0.50, 0.45),
    }
    return pd.DataFrame(
        [
            {
                "surrogate": surrogate,
                "recipe": recipe,
                "mean_cka": mean_cka,
                "dbs": dbs,
            }
            for surrogate, (mean_cka, dbs) in similarities.items()
            for recipe in ("r1", "r2")
        ]
    )


def _repeat_outcome(
    *,
    surrogate: str,
    recipe: str,
    source: str,
    total: int,
    target_benign: int,
    original_pred: int = 1,
) -> list[dict[str, int | str]]:
    """Build repeated synthetic prediction-only records with a target-benign count."""
    return [
        _annotated_record(
            surrogate=surrogate,
            recipe=recipe,
            source=source,
            original_pred=original_pred,
            perturbed_pred=0 if index < target_benign else 1,
        )
        for index in range(total)
    ]


def _multi_surrogate_records() -> list[dict[str, int | str]]:
    """Build a synthetic pool with a removable known-source contribution."""
    safe_source = "deepset/prompt-injections"
    known_source = target_audit.KNOWN_TARGET_TRAINING_SOURCE
    return [
        *_repeat_outcome(
            surrogate="s1",
            recipe="r1",
            source=known_source,
            total=1,
            target_benign=1,
        ),
        *_repeat_outcome(
            surrogate="s1",
            recipe="r2",
            source=safe_source,
            total=1,
            target_benign=0,
        ),
        *_repeat_outcome(
            surrogate="s2",
            recipe="r1",
            source=safe_source,
            total=1,
            target_benign=1,
        ),
        *_repeat_outcome(
            surrogate="s2",
            recipe="r2",
            source=safe_source,
            total=1,
            target_benign=1,
        ),
        *_repeat_outcome(
            surrogate="s3",
            recipe="r1",
            source=safe_source,
            total=5,
            target_benign=1,
        ),
        *_repeat_outcome(
            surrogate="s3",
            recipe="r2",
            source=safe_source,
            total=5,
            target_benign=1,
        ),
        *_repeat_outcome(
            surrogate="s4",
            recipe="r1",
            source=safe_source,
            total=10,
            target_benign=3,
        ),
        *_repeat_outcome(
            surrogate="s4",
            recipe="r2",
            source=safe_source,
            total=10,
            target_benign=3,
        ),
        *_repeat_outcome(
            surrogate="s5",
            recipe="r1",
            source=safe_source,
            total=5,
            target_benign=3,
            original_pred=0,
        ),
        *_repeat_outcome(
            surrogate="s5",
            recipe="r2",
            source=safe_source,
            total=5,
            target_benign=3,
            original_pred=0,
        ),
    ]


def test_build_full_grid_cells_pads_zero_successes_with_null_rates() -> None:
    """Master keys, rather than records, define every persisted audit cell."""
    master = _complete_master_grid().iloc[:2].copy()
    records = [_annotated_record(surrogate="s1", recipe="r1")]

    cells = target_audit.build_full_grid_cells(
        records,
        master,
        benign_label=0,
        injection_label=1,
    )

    assert len(cells) == 2
    zero_success = cells.loc[cells["recipe"] == "r2"].iloc[0]
    for field in (
        "source_successful",
        "target_original_benign",
        "target_original_injection",
        "target_perturbed_benign",
        "true_target_flips",
    ):
        assert zero_success[field] == 0
    assert zero_success["conditional_target_benign_rate"] is None
    assert zero_success["true_target_flip_rate"] is None
    assert not any(column.endswith(("_master", "_x", "_y")) for column in cells.columns)
    assert json.dumps(cells.to_dict("records"), allow_nan=False)


def test_audit_summary_uses_prejoined_cells_without_a_second_master_argument() -> None:
    """Summary analysis consumes the already complete master-keyed cell grid."""
    master = _complete_master_grid()
    records = _multi_surrogate_records()
    cells = target_audit.build_full_grid_cells(
        records,
        master,
        benign_label=0,
        injection_label=1,
    )

    summary = target_audit.build_summary_from_records(
        records,
        cells,
        {"M1": ["s1", "s2"], "M2": ["s3", "s4"]},
        {"ablation": {"n_permutations": 10, "alpha": 0.05}},
        seed=19,
        excluded_source=target_audit.KNOWN_TARGET_TRAINING_SOURCE,
        benign_label=0,
        injection_label=1,
        unique_text_stats={"unique_originals": 0, "unique_perturbations": 0, "unique_texts": 0},
        baseline_counts={"target_original_benign": 0, "target_original_injection": 0},
    )

    assert json.dumps(summary, allow_nan=False)


def test_full_and_source_excluded_analyses_are_surrogate_level_for_both_outcomes() -> None:
    """Both cohorts use one macro row per surrogate before association and M1/M2 tests."""
    master = _complete_master_grid()
    records = _multi_surrogate_records()
    cells = target_audit.build_full_grid_cells(
        records,
        master,
        benign_label=0,
        injection_label=1,
    )

    summary = target_audit.build_summary_from_records(
        records,
        cells,
        {"M1": ["s1", "s2"], "M2": ["s3", "s4"]},
        {"ablation": {"n_permutations": 10, "alpha": 0.05}},
        seed=23,
        excluded_source=target_audit.KNOWN_TARGET_TRAINING_SOURCE,
        benign_label=0,
        injection_label=1,
        unique_text_stats={"unique_originals": 0, "unique_perturbations": 0, "unique_texts": 0},
        baseline_counts={"target_original_benign": 0, "target_original_injection": 0},
    )

    for analysis_name in ("full_cohort", "known_source_excluded_sensitivity"):
        analysis = summary[analysis_name]
        macro = analysis["surrogate_macro"]
        assert macro["analysis_unit"] == "one row per surrogate"
        assert macro["row_count"] == 5
        assert len(macro["rows"]) == 5
        assert all("recipe" not in row for row in macro["rows"])
        rows = {row["surrogate"]: row for row in macro["rows"]}
        assert rows["s5"]["true_target_flip_rate_macro_mean"] is None
        assert rows["s5"]["true_target_flip_rate_defined_cell_count"] == 0

        associations = analysis["cka_dbs_association"]
        for outcome in ("conditional_target_benign_rate", "true_target_flip_rate"):
            for feature in ("mean_cka", "dbs"):
                block = associations[outcome][feature]
                assert block["designed_pool"]["unit"] == "one row per surrogate"
                exchangeability = block["exchangeability_null"]
                assert exchangeability["exact"] is True
                assert (
                    exchangeability["p_value_label"]
                    == "exact enumeration under the exchangeability null for the designed pool"
                )

        assert (
            associations["conditional_target_benign_rate"]["mean_cka"]["exchangeability_null"]["n"]
            == 5
        )
        assert associations["true_target_flip_rate"]["mean_cka"]["exchangeability_null"]["n"] == 4

        ablation = analysis["selection_ablation"]
        for outcome in ("conditional_target_benign_rate", "true_target_flip_rate"):
            test = ablation[outcome]["one_sided_exchangeability_null"]
            assert test["exact"] is True
            assert 0.0 <= test["mean_p_value"] <= 1.0
            assert (
                test["p_value_label"]
                == "exact enumeration under the exchangeability null for the designed M1/M2 pool"
            )

    excluded = summary["known_source_excluded_sensitivity"]
    assert excluded["excluded_source"] == target_audit.KNOWN_TARGET_TRAINING_SOURCE
    assert excluded["source_successful"] == 43
    excluded_s1 = {row["surrogate"]: row for row in excluded["surrogate_macro"]["rows"]}["s1"]
    assert excluded_s1["conditional_target_benign_rate_macro_mean"] == pytest.approx(0.0)
    assert excluded_s1["conditional_target_benign_rate_defined_cell_count"] == 1
