"""Behavioral tests for saved-pool conditional rates and fixed-trial yields."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from transfer_risk.lib.pool_sensitivity import build_pool_sensitivity

_MACRO = "macro_conditional_verified_flip_rate"
_YIELD = "source_success_gated_verified_flip_yield"
_ROOT = Path(__file__).resolve().parents[2]


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return unequal conditional cohorts under identical attempted trial budgets."""
    records = []
    for name, denominators, flips in (
        ("a", (10, 2), (1, 1)),
        ("b", (10, 10), (2, 2)),
        ("c", (10, 10), (3, 3)),
        ("d", (10, 10), (4, 8)),
    ):
        for recipe, denominator, numerator in zip(("r1", "r2"), denominators, flips, strict=True):
            records.append(
                {
                    "surrogate": name,
                    "recipe": recipe,
                    "source_successful": 10,
                    "target_original_benign": 10 - denominator,
                    "target_original_injection": denominator,
                    "target_perturbed_benign": 10 - denominator + numerator,
                    "true_target_flips": numerator,
                }
            )
    similarity = pd.DataFrame({"surrogate": ["a", "b", "c", "d"], "mean_cka": [0.1, 0.2, 0.3, 0.4]})
    return pd.DataFrame(records), similarity


def _report(cells: pd.DataFrame, similarity: pd.DataFrame, **options: Any) -> dict[str, Any]:
    """Build the small fixture with explicit budgets, recipes, and exclusions."""
    arguments = {
        "attempts_per_cell": 20,
        "recipes": ["r1", "r2"],
        "excluded_surrogates": ["a"],
        "selection_size": 2,
        **options,
    }
    return build_pool_sensitivity(cells, similarity, **arguments)


def test_macro_rate_and_fixed_trial_yield_keep_different_denominators() -> None:
    """Weighting by successful baselines must not replace the attempted budget."""
    cells, similarity = _inputs()
    report = _report(cells, similarity)
    rows = {row["surrogate"]: row for row in report["surrogate_rows"]}
    assert rows["a"]["macro_conditional_verified_flip_rate"] == pytest.approx(0.3)
    assert rows["a"]["pooled_conditional_verified_flip_rate"] == pytest.approx(2 / 12)
    assert rows["a"]["source_success_gated_verified_flip_yield"] == pytest.approx(2 / 40)
    assert rows["a"]["attempted"] == 40
    assert rows["a"]["not_source_successful"] == 20
    assert rows["a"]["defined_conditional_recipe_count"] == 2
    assert report["provenance"]["n_cells"] == 8
    assert report["pools"]["all_surrogates"]["counts"]["true_target_flips"] == 24
    assert report["pools"]["all_surrogates"]["counts"]["attempted"] == 160


def test_tied_outcomes_use_average_ranks_and_label_the_p_value_method() -> None:
    """Recipe means can tie even when source counts and CKA values differ."""
    report = _report(*_inputs())
    full = report["pools"]["all_surrogates"]["associations"]
    assert full[_MACRO]["rho"] == pytest.approx(0.6324555320336759)
    assert full[_MACRO]["exact"] is False
    assert full[_MACRO]["p_value_method"] == "normal_approximation"
    assert full[_YIELD]["rho"] == pytest.approx(1.0)
    assert full[_YIELD]["two_sided_p"] == pytest.approx(1 / 12)
    assert full[_YIELD]["exact"] is True
    assert full[_YIELD]["p_value_method"] == "exact_enumeration"
    reduced = report["pools"]["surrogate_excluded"]
    assert reduced["surrogates"] == ["b", "c", "d"]
    assert reduced["counts"]["attempted"] == 120
    assert reduced["counts"]["true_target_flips"] == 22
    assert reduced["associations"][_MACRO]["rho"] == pytest.approx(1.0)
    assert reduced["associations"][_MACRO]["two_sided_p"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("zero_both", [False, True])
def test_zero_target_positive_cells_preserve_undefined_conditional_rates(
    *,
    zero_both: bool,
) -> None:
    """A zero baseline is omitted from the recipe mean and remains in trial yield."""
    cells, similarity = _inputs()
    indices = cells.index[cells["surrogate"].eq("a")]
    changed = indices if zero_both else indices[:1]
    cells.loc[changed, "target_original_injection"] = 0
    cells.loc[changed, "target_original_benign"] = 10
    cells.loc[changed, "target_perturbed_benign"] = 10
    cells.loc[changed, "true_target_flips"] = 0
    report = _report(cells, similarity)
    row = report["surrogate_rows"][0]
    assert row["surrogate"] == "a"
    assert row["attempted"] == 40
    assert row["defined_conditional_recipe_count"] == (0 if zero_both else 1)
    if zero_both:
        assert row[_MACRO] is None
        assert row[_YIELD] == 0.0
    else:
        assert row[_MACRO] == pytest.approx(0.5)
        assert row[_YIELD] == pytest.approx(1 / 40)
    association = report["pools"]["all_surrogates"]["associations"][_MACRO]
    assert association["n"] == (3 if zero_both else 4)
    assert association["undefined_surrogates"] == (["a"] if zero_both else [])
    assert association["complete_case_pool"] is (not zero_both)
    assert report["pools"]["all_surrogates"]["associations"][_YIELD]["n"] == 4
    cell_row = next(
        row for row in report["cell_rows"] if row["surrogate"] == "a" and row["recipe"] == "r1"
    )
    assert cell_row["true_target_flip_rate"] is None
    assert cell_row[_YIELD] == 0.0


def test_zero_source_success_cell_remains_in_complete_trial_grid() -> None:
    """A recipe with no source successes contributes zero verified trial yield."""
    cells, similarity = _inputs()
    for field in (
        "source_successful",
        "target_original_benign",
        "target_original_injection",
        "target_perturbed_benign",
        "true_target_flips",
    ):
        cells.loc[0, field] = 0
    cells["true_target_flip_rate"] = cells["true_target_flips"] / cells["target_original_injection"]
    report = _report(cells, similarity)
    assert report["provenance"]["n_cells"] == 8
    assert report["cell_rows"][0]["attempted"] == 20
    assert report["cell_rows"][0][_YIELD] == 0.0
    assert report["surrogate_rows"][0][_MACRO] == pytest.approx(0.5)


def test_enumerating_selections_keeps_trial_budget_and_counts_auditable() -> None:
    """All six pairs use 80 trials, regardless of their successful-source counts."""
    report = _report(*_inputs())
    selection = report["pools"]["all_surrogates"]["selection"]
    assert selection["n_subsets"] == 6
    assert selection["attempts_per_subset"] == 80
    assert selection["selected_surrogates"] == ["c", "d"]
    subsets = {tuple(row["surrogates"]): row for row in selection["subsets"]}
    assert subsets[("a", "b")]["true_target_flips"] == 6
    assert subsets[("a", "b")][_YIELD] == pytest.approx(6 / 80)
    assert subsets[("c", "d")]["true_target_flips"] == 18
    assert subsets[("c", "d")][_YIELD] == pytest.approx(18 / 80)
    assert all(row["attempted"] == 80 for row in subsets.values())
    comparison = selection["comparisons"][_YIELD]
    assert comparison["n_strictly_higher"] == 0
    assert comparison["n_equal"] == 1
    assert comparison["n_strictly_lower"] == 5
    assert comparison["mean"] == pytest.approx(0.15)
    assert comparison["selected_minus_mean"] == pytest.approx(0.075)
    assert comparison["fraction_at_or_above"] == pytest.approx(1 / 6)
    assert "p_value" not in comparison


def test_cka_cutoff_ties_are_explicit_and_deterministic() -> None:
    """An arbitrary input row order cannot decide a tied CKA selection."""
    cells, similarity = _inputs()
    similarity.loc[similarity["surrogate"].isin(["b", "c"]), "mean_cka"] = 0.3
    report = _report(cells, similarity)
    selection = report["pools"]["all_surrogates"]["selection"]
    assert selection["selected_surrogates"] == ["b", "d"]
    assert selection["cka_cutoff_tied_surrogates"] == ["b", "c"]
    assert selection["tie_break"] == "ascending surrogate identifier"
    assert report == _report(cells.iloc[::-1], similarity.iloc[::-1], recipes=["r2", "r1"])


def test_constant_cka_marks_associations_and_influence_as_not_estimable() -> None:
    """Descriptive summaries remain valid when rank correlation has no variance."""
    cells, similarity = _inputs()
    similarity["mean_cka"] = 0.2
    report = _report(cells, similarity)
    pool = report["pools"]["all_surrogates"]
    for outcome in (_MACRO, _YIELD):
        assert pool["associations"][outcome]["status"] == "not_estimable"
        assert pool["associations"][outcome]["rho"] is None
        assert pool["associations"][outcome]["two_sided_p"] is None
        assert "constant" in pool["associations"][outcome]["reason"]
        assert pool["leave_one_out_summary"][outcome]["rho_min"] is None
        assert pool["leave_one_out_summary"][outcome]["rho_max"] is None
        assert all(row["associations"][outcome]["rho"] is None for row in pool["leave_one_out"])
    json.dumps(report, allow_nan=False)


def test_fewer_than_two_conditional_models_are_not_estimable() -> None:
    """Yield remains comparable when conditional complete cases are insufficient."""
    cells, similarity = _inputs()
    changed = cells["surrogate"].ne("d")
    cells.loc[changed, "target_original_injection"] = 0
    cells.loc[changed, "target_original_benign"] = 10
    cells.loc[changed, "target_perturbed_benign"] = 10
    cells.loc[changed, "true_target_flips"] = 0
    report = _report(cells, similarity)
    association = report["pools"]["all_surrogates"]["associations"][_MACRO]
    assert association["status"] == "not_estimable"
    assert association["n"] == 1
    assert association["included_surrogates"] == ["d"]
    assert association["rho"] is None
    selection = report["pools"]["all_surrogates"]["selection"]
    assert all(row[_MACRO] is None for row in selection["subsets"])
    assert selection["comparisons"][_MACRO]["n_subsets_with_defined_outcome"] == 0


def _assert_native_values(value: Any) -> None:
    """Check that the artifact requires no NumPy or pandas JSON encoder."""
    if isinstance(value, dict):
        assert all(type(key) is str for key in value)
        for item in value.values():
            _assert_native_values(item)
    elif isinstance(value, list):
        for item in value:
            _assert_native_values(item)
    else:
        assert type(value) in (str, int, float, bool, type(None))


def test_output_is_native_order_invariant_and_aggregate_only() -> None:
    """Arbitrary private columns and dataframe order cannot leak into an artifact."""
    cells, similarity = _inputs()
    cells["text"] = "PRIVATE_PROMPT_FIXTURE"
    similarity["checkpoint_path"] = "PRIVATE_CHECKPOINT_FIXTURE"
    cells_before = deepcopy(cells)
    similarity_before = deepcopy(similarity)
    report = _report(cells, similarity)
    _assert_native_values(report)
    encoded = json.dumps(report, allow_nan=False)
    assert "PRIVATE_" not in encoded
    assert report == _report(cells.iloc[::-1], similarity.iloc[::-1])
    pd.testing.assert_frame_equal(cells, cells_before)
    pd.testing.assert_frame_equal(similarity, similarity_before)
    assert len(report["provenance"]["audit_counts_sha256"]) == 64
    assert len(report["provenance"]["similarity_values_sha256"]) == 64


@pytest.mark.parametrize(
    "column",
    [
        "surrogate",
        "recipe",
        "source_successful",
        "target_original_benign",
        "target_original_injection",
        "target_perturbed_benign",
        "true_target_flips",
    ],
)
def test_missing_aggregate_columns_fail(column: str) -> None:
    """Dropping a denominator field cannot silently change an estimand."""
    cells, similarity = _inputs()
    with pytest.raises(ValueError, match=r"missing.*columns"):
        _report(cells.drop(columns=column), similarity)


@pytest.mark.parametrize("column", ["surrogate", "mean_cka"])
def test_missing_similarity_columns_fail(column: str) -> None:
    cells, similarity = _inputs()
    with pytest.raises(ValueError, match=r"missing.*columns"):
        _report(cells, similarity.drop(columns=column))


def test_duplicate_dataframe_columns_fail() -> None:
    cells, similarity = _inputs()
    with pytest.raises(ValueError, match=r"duplicate.*columns"):
        _report(pd.concat([cells, cells[["true_target_flips"]]], axis=1), similarity)


@pytest.mark.parametrize("malformation", ["missing_cell", "duplicate_cell", "extra_recipe"])
def test_incomplete_or_duplicate_grid_fails(malformation: str) -> None:
    """Missing saved cells cannot be silently imputed as zero target outcomes."""
    cells, similarity = _inputs()
    if malformation == "missing_cell":
        cells = cells.iloc[1:]
    elif malformation == "duplicate_cell":
        cells = pd.concat([cells, cells.iloc[:1]], ignore_index=True)
    else:
        extra = cells.iloc[:1].copy()
        extra["recipe"] = "r3"
        cells = pd.concat([cells, extra], ignore_index=True)
    with pytest.raises(ValueError, match=r"grid|duplicate.*cell"):
        _report(cells, similarity)


def test_missing_whole_recipe_fails_against_configured_recipe_list() -> None:
    cells, similarity = _inputs()
    with pytest.raises(ValueError, match="grid"):
        _report(cells.loc[cells["recipe"].eq("r1")], similarity)


def test_pool_mismatch_between_similarity_and_cells_fails() -> None:
    cells, similarity = _inputs()
    with pytest.raises(ValueError, match="same surrogate pool"):
        _report(cells.loc[cells["surrogate"].ne("a")], similarity)


@pytest.mark.parametrize("field", ["surrogate", "recipe"])
@pytest.mark.parametrize("value", [None, "", " a", 2])
def test_invalid_cell_identifiers_fail(field: str, value: Any) -> None:
    cells, similarity = _inputs()
    cells[field] = cells[field].astype(object)
    cells.loc[0, field] = value
    with pytest.raises(ValueError, match="identifier"):
        _report(cells, similarity)


@pytest.mark.parametrize("value", [-1, 1.5, float("nan"), float("inf"), True, "1"])
def test_noninteger_or_nonfinite_counts_fail(value: Any) -> None:
    cells, similarity = _inputs()
    cells["true_target_flips"] = cells["true_target_flips"].astype(object)
    cells.loc[0, "true_target_flips"] = value
    with pytest.raises(ValueError, match="count"):
        _report(cells, similarity)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_successful", 21),
        ("source_successful", 9),
        ("true_target_flips", 11),
        ("target_perturbed_benign", 0),
        ("target_perturbed_benign", 7),
    ],
)
def test_inconsistent_audit_marginals_fail(field: str, value: int) -> None:
    """Impossible joint target transitions cannot become numerical evidence."""
    cells, similarity = _inputs()
    cells.loc[0, field] = value
    with pytest.raises(ValueError, match=r"counts|attempted"):
        _report(cells, similarity)


@pytest.mark.parametrize("column", ["true_target_flip_rate", "conditional_target_benign_rate"])
def test_saved_rate_inconsistent_with_counts_fails(column: str) -> None:
    cells, similarity = _inputs()
    denominator = (
        cells["target_original_injection"]
        if column == "true_target_flip_rate"
        else cells["source_successful"]
    )
    numerator = (
        cells["true_target_flips"]
        if column == "true_target_flip_rate"
        else cells["target_perturbed_benign"]
    )
    cells[column] = numerator / denominator
    cells.loc[0, column] = 0.99
    with pytest.raises(ValueError, match=r"rate.*counts"):
        _report(cells, similarity)


def test_saved_rate_with_missing_nonzero_denominator_fails() -> None:
    cells, similarity = _inputs()
    cells["true_target_flip_rate"] = cells["true_target_flips"] / cells["target_original_injection"]
    cells.loc[0, "true_target_flip_rate"] = float("nan")
    with pytest.raises(ValueError, match=r"rate.*counts"):
        _report(cells, similarity)


def test_cell_cka_inconsistent_with_similarity_row_fails() -> None:
    cells, similarity = _inputs()
    cells["mean_cka"] = cells["surrogate"].map(similarity.set_index("surrogate")["mean_cka"])
    cells.loc[0, "mean_cka"] = 0.9
    with pytest.raises(ValueError, match=r"mean_cka.*inconsistent"):
        _report(cells, similarity)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_malformed_similarity_values_fail(value: Any) -> None:
    cells, similarity = _inputs()
    similarity["mean_cka"] = similarity["mean_cka"].astype(object)
    similarity.loc[0, "mean_cka"] = value
    with pytest.raises(ValueError, match="mean_cka"):
        _report(cells, similarity)


def test_duplicate_similarity_rows_fail() -> None:
    cells, similarity = _inputs()
    with pytest.raises(ValueError, match="duplicate surrogate"):
        _report(cells, pd.concat([similarity, similarity.iloc[:1]], ignore_index=True))


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_attempted_budget_fails(value: Any) -> None:
    with pytest.raises(ValueError, match="attempts_per_cell"):
        _report(*_inputs(), attempts_per_cell=value)


@pytest.mark.parametrize("recipes", [[], ["r1", "r1"], ["r1", ""]])
def test_malformed_recipe_configuration_fails(recipes: list[str]) -> None:
    with pytest.raises(ValueError, match="recipes"):
        _report(*_inputs(), recipes=recipes)


@pytest.mark.parametrize("excluded", [["unknown"], ["a", "a"], ["a", "b", "c"]])
def test_malformed_exclusion_fails(excluded: list[str]) -> None:
    with pytest.raises(ValueError, match=r"excluded|retained"):
        _report(*_inputs(), excluded_surrogates=excluded)


@pytest.mark.parametrize("value", [0, 5, True, 1.5])
def test_invalid_selection_size_fails(value: Any) -> None:
    with pytest.raises(ValueError, match="selection_size"):
        _report(*_inputs(), selection_size=value)


def test_no_exclusion_preserves_only_the_complete_pool() -> None:
    report = _report(*_inputs(), excluded_surrogates=[])
    assert list(report["pools"]) == ["all_surrogates"]


def test_public_saved_pool_exposes_weaker_yield_association_and_selection() -> None:
    """Conditional transfer cannot stand in for a fixed-trial selection objective."""
    cells = pd.read_csv(_ROOT / "docs/artifacts/target_audit_cells.csv")
    similarity = pd.read_csv(_ROOT / "docs/artifacts/surrogate_summary.csv")
    report = build_pool_sensitivity(
        cells,
        similarity,
        attempts_per_cell=191,
        recipes=["deepwordbug", "pwws", "textfooler", "bae", "bert-attack"],
    )
    full = report["pools"]["all_surrogates"]
    assert full["counts"]["true_target_flips"] == 1054
    assert full["counts"]["target_original_injection"] == 4009
    assert full["counts"]["attempted"] == 9550
    assert full["associations"][_MACRO]["rho"] == pytest.approx(0.8303030303030303)
    assert full["associations"][_MACRO]["two_sided_p"] == pytest.approx(0.004710648148148148)
    assert full["associations"][_YIELD]["rho"] == pytest.approx(0.5393939393939394)
    assert full["associations"][_YIELD]["two_sided_p"] == pytest.approx(0.11385416666666667)
    assert full["associations"][_YIELD]["exact"] is True
    reduced = report["pools"]["surrogate_excluded"]
    assert reduced["counts"]["true_target_flips"] == 998
    assert reduced["counts"]["attempted"] == 8595
    assert reduced["associations"][_MACRO]["rho"] == pytest.approx(0.8)
    assert reduced["associations"][_MACRO]["two_sided_p"] == pytest.approx(0.01382826278659612)
    assert reduced["associations"][_YIELD]["rho"] == pytest.approx(0.4)
    assert reduced["associations"][_YIELD]["two_sided_p"] == pytest.approx(0.2911926807760141)
    influences = {row["omitted_surrogate"]: row for row in full["leave_one_out"]}
    assert influences["bilstm-attention"]["associations"][_YIELD]["rho"] == pytest.approx(0.4)
    assert influences["deepset-deberta-injection"]["associations"][_YIELD]["rho"] == pytest.approx(
        0.7666666666666667
    )
    assert influences["deepset-deberta-injection"]["associations"][_YIELD][
        "rho_change_from_pool"
    ] == pytest.approx(0.2272727272727273)
    assert full["leave_one_out_summary"][_MACRO]["rho_min"] == pytest.approx(0.7666666666666667)
    assert full["leave_one_out_summary"][_MACRO]["rho_max"] == pytest.approx(0.8666666666666667)
    assert full["leave_one_out_summary"][_YIELD]["rho_min"] == pytest.approx(0.4)
    assert full["leave_one_out_summary"][_YIELD]["rho_max"] == pytest.approx(0.7666666666666667)
    selection = full["selection"]
    assert selection["n_subsets"] == 120
    assert selection["attempts_per_subset"] == 2865
    assert selection["selected_subset"]["true_target_flips"] == 424
    assert selection["selected_subset"][_YIELD] == pytest.approx(424 / 2865)
    comparison = selection["comparisons"][_YIELD]
    assert comparison["n_strictly_higher"] == 11
    assert comparison["n_equal"] == 1
    assert comparison["n_strictly_lower"] == 108
    assert comparison["mean"] == pytest.approx(316.2 / 2865)
    assert comparison["max"] == pytest.approx(507 / 2865)
    assert comparison["best_selections"] == [
        ["deberta-base-ft-seed", "deberta-small-pi-v2", "roberta-base-ft"]
    ]
    assert reduced["selection"]["n_subsets"] == 84
