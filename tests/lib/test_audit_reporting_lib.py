"""Tests for aggregate-only public target-audit reporting helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd
import pytest

from transfer_risk.lib import audit_reporting

_SURROGATES = tuple(f"s{index}" for index in range(10))
_RECIPES = tuple(f"recipe-{index}" for index in range(5))
_M1 = ("s0", "s1", "s2")
_M2 = ("s3", "s4", "s5")


def _cells() -> pd.DataFrame:
    """Build an in-memory complete aggregate-only audit grid."""
    return pd.DataFrame.from_records(
        [
            {
                "surrogate": surrogate,
                "recipe": recipe,
                "source_successful": 20,
                "target_original_benign": 4,
                "target_original_injection": 16,
                "target_perturbed_benign": 6,
                "true_target_flips": 4,
                "conditional_target_benign_rate": 0.3,
                "true_target_flip_rate": 0.25 + (surrogate_index * 0.01),
                "mean_cka": 0.2 + (surrogate_index * 0.03),
                "dbs": 0.1 + (surrogate_index * 0.02),
                "original": "private original",
                "perturbed": "private perturbation",
            }
            for surrogate_index, surrogate in enumerate(_SURROGATES)
            for recipe in _RECIPES
        ]
    )


def _selection() -> dict[str, list[str]]:
    """Return a valid fixed M1/M2 selection."""
    return {"M1": list(_M1), "M2": list(_M2)}


def _summary() -> dict[str, Any]:
    """Build a complete non-text summary needed by both public plot datasets."""
    macro_rows = [
        {
            "surrogate": surrogate,
            "mean_cka": 0.2 + (index * 0.03),
            "true_target_flip_rate_macro_mean": 0.25 + (index * 0.01),
        }
        for index, surrogate in enumerate(_SURROGATES)
    ]
    return {
        "full_cohort": {
            "surrogate_macro": {"rows": macro_rows},
            "cka_dbs_association": {
                "true_target_flip_rate": {
                    "mean_cka": {
                        "exchangeability_null": {
                            "exact": True,
                            "rho": 0.7,
                            "two_sided_p": 0.02,
                            "n": 10,
                        }
                    }
                }
            },
            "selection_ablation": {
                "true_target_flip_rate": {
                    "one_sided_exchangeability_null": {
                        "exact": True,
                        "m1_mean": 0.4,
                        "m2_mean": 0.2,
                        "mean_diff_pp": 20.0,
                        "mean_p_value": 0.05,
                        "n_m1": 3,
                        "n_m2": 3,
                    }
                }
            },
        },
        "known_source_excluded_sensitivity": {},
        "membership": _selection(),
    }


def test_public_audit_cells_require_every_aggregate_column() -> None:
    """A public cell export cannot silently omit a required aggregate."""
    cells = _cells().drop(columns="dbs")

    with pytest.raises(ValueError, match="missing required columns"):
        audit_reporting.export_public_target_audit_cells(cells)


def test_public_audit_cells_require_the_complete_fifty_cell_grid() -> None:
    """A partial public audit grid is rejected rather than published as complete."""
    cells = _cells().iloc[:-1]

    with pytest.raises(ValueError, match="exactly 50 rows"):
        audit_reporting.export_public_target_audit_cells(cells)


def test_public_audit_sources_require_source_provenance() -> None:
    """Source-level exports require the aggregate source identifier."""
    sources = _cells().drop(columns="source", errors="ignore")

    with pytest.raises(ValueError, match="missing required columns"):
        audit_reporting.export_public_target_audit_sources(sources)


def test_safe_audit_summary_requires_all_finalized_sections() -> None:
    """A public summary must retain cohort, sensitivity, and membership sections."""
    with pytest.raises(ValueError, match="missing required sections"):
        audit_reporting.publish_safe_audit_summary({"full_cohort": {}})


def test_safe_audit_summary_rejects_nonfinite_numeric_values() -> None:
    """Public JSON summaries cannot encode non-finite outcome values."""
    summary = _summary()
    summary["full_cohort"]["headline_rate"] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        audit_reporting.publish_safe_audit_summary(summary)


def test_safe_audit_summary_rejects_raw_prompt_field_suffixes() -> None:
    """Raw text is blocked even when it is stored under a noncanonical prompt key."""
    summary = _summary()
    summary["membership"]["attack_prompt"] = "private prompt"

    with pytest.raises(ValueError, match="raw-text field"):
        audit_reporting.publish_safe_audit_summary(summary)


def test_safe_audit_summary_rejects_non_json_values() -> None:
    """Public JSON summaries reject unsupported in-memory Python values."""
    summary = _summary()
    summary["membership"]["unsafe_ids"] = {1, 2}

    with pytest.raises(TypeError, match="unsupported JSON value"):
        audit_reporting.publish_safe_audit_summary(summary)


def test_true_flip_scatter_requires_saved_exact_association() -> None:
    """Scatter publication requires the saved exact CKA association structure."""
    summary = _summary()
    del summary["full_cohort"]["cka_dbs_association"]

    with pytest.raises(ValueError, match="missing mapping"):
        audit_reporting.prepare_true_flip_scatter_data(_cells(), summary)


def test_true_flip_scatter_rejects_undefined_recipe_rates() -> None:
    """Scatter data rejects a cell whose true-flip rate has no defined denominator."""
    cells = _cells()
    cells.loc[cells.index[0], "true_target_flip_rate"] = None

    with pytest.raises(ValueError, match=r"non-finite 'true_target_flip_rate'"):
        audit_reporting.prepare_true_flip_scatter_data(cells, _summary())


def test_true_flip_scatter_rejects_malformed_macro_rows() -> None:
    """Scatter data requires mapping-shaped surrogate macro rows."""
    summary = _summary()
    summary["full_cohort"]["surrogate_macro"]["rows"] = ["not a macro row"] * 10

    with pytest.raises(ValueError, match="macro rows must be mappings"):
        audit_reporting.prepare_true_flip_scatter_data(_cells(), summary)


def test_true_flip_scatter_rejects_nonmapping_macro_aggregate() -> None:
    """Nested macro aggregates must remain mappings before their rows are read."""
    summary = _summary()
    summary["full_cohort"]["surrogate_macro"] = []

    with pytest.raises(ValueError, match="missing sequence"):
        audit_reporting.prepare_true_flip_scatter_data(_cells(), summary)


def test_true_flip_scatter_rejects_duplicate_macro_surrogates() -> None:
    """A public scatter requires one aggregate macro row per surrogate."""
    summary = _summary()
    rows = summary["full_cohort"]["surrogate_macro"]["rows"]
    rows[-1]["surrogate"] = rows[0]["surrogate"]

    with pytest.raises(ValueError, match="exactly ten unique"):
        audit_reporting.prepare_true_flip_scatter_data(_cells(), summary)


def test_true_flip_ablation_requires_matching_saved_membership() -> None:
    """A plot cannot relabel groups when the summary membership changed."""
    summary = _summary()
    summary["membership"]["M1"] = ["s9", "s1", "s2"]

    with pytest.raises(ValueError, match="membership differs"):
        audit_reporting.prepare_true_flip_ablation_data(summary, _selection())


def test_true_flip_ablation_requires_three_surrogates_per_group() -> None:
    """The exact six-surrogate ablation rejects a partial saved selection."""
    selection = {"M1": ["s0", "s1"], "M2": list(_M2)}
    summary = _summary()
    summary["membership"] = deepcopy(selection)

    with pytest.raises(ValueError, match="exactly three M1 and three M2"):
        audit_reporting.prepare_true_flip_ablation_data(summary, selection)


def test_true_flip_ablation_requires_exact_saved_statistics() -> None:
    """Ablation plot annotations cannot be derived from non-exact statistics."""
    summary = _summary()
    summary["full_cohort"]["selection_ablation"]["true_target_flip_rate"][
        "one_sided_exchangeability_null"
    ]["exact"] = False

    with pytest.raises(ValueError, match="must use exact enumeration"):
        audit_reporting.prepare_true_flip_ablation_data(summary, _selection())


def test_true_flip_scatter_requires_exact_association_statistics() -> None:
    """Scatter annotations cannot be built from non-exact association statistics."""
    summary = _summary()
    summary["full_cohort"]["cka_dbs_association"]["true_target_flip_rate"]["mean_cka"][
        "exchangeability_null"
    ]["exact"] = False

    with pytest.raises(ValueError, match="must use exact enumeration"):
        audit_reporting.prepare_true_flip_scatter_data(_cells(), summary)


def test_true_flip_scatter_rejects_malformed_association_mapping() -> None:
    """The saved exchangeability-null payload must remain a mapping."""
    summary = _summary()
    summary["full_cohort"]["cka_dbs_association"]["true_target_flip_rate"]["mean_cka"][
        "exchangeability_null"
    ] = "not a mapping"

    with pytest.raises(ValueError, match="must be a mapping"):
        audit_reporting.prepare_true_flip_scatter_data(_cells(), summary)


def test_true_flip_scatter_requires_macro_row_sequence() -> None:
    """Macro outcomes must be a real sequence rather than an arbitrary text field."""
    summary = _summary()
    summary["full_cohort"]["surrogate_macro"]["rows"] = "not a sequence of rows"

    with pytest.raises(ValueError, match="must be a sequence"):
        audit_reporting.prepare_true_flip_scatter_data(_cells(), summary)


def test_true_flip_scatter_requires_integral_association_sample_size() -> None:
    """The exact association's designed-pool count must be an integer."""
    summary = _summary()
    summary["full_cohort"]["cka_dbs_association"]["true_target_flip_rate"]["mean_cka"][
        "exchangeability_null"
    ]["n"] = 10.5

    with pytest.raises(ValueError, match="must be an integer"):
        audit_reporting.prepare_true_flip_scatter_data(_cells(), summary)


def test_true_flip_ablation_requires_both_saved_selection_groups() -> None:
    """Ablation plotting requires an explicit M1 group as well as M2."""
    with pytest.raises(ValueError, match=r"missing 'M1'"):
        audit_reporting.prepare_true_flip_ablation_data(_summary(), {"M2": list(_M2)})


def test_true_flip_ablation_rejects_duplicate_selection_members() -> None:
    """A saved group cannot repeat a surrogate when plotting group evidence."""
    selection = {"M1": ["s0", "s0", "s2"], "M2": list(_M2)}

    with pytest.raises(ValueError, match="must contain unique names"):
        audit_reporting.prepare_true_flip_ablation_data(_summary(), selection)


def test_true_flip_ablation_requires_macro_rows_for_selected_surrogates() -> None:
    """Every saved M1/M2 member needs one corresponding macro outcome row."""
    summary = _summary()
    summary["full_cohort"]["surrogate_macro"]["rows"][0]["surrogate"] = "unselected"

    with pytest.raises(ValueError, match="has no macro row for selected surrogate"):
        audit_reporting.prepare_true_flip_ablation_data(summary, _selection())


def test_true_flip_ablation_requires_matching_saved_group_counts() -> None:
    """Ablation statistics must report the same group sizes as the saved selection."""
    summary = _summary()
    summary["full_cohort"]["selection_ablation"]["true_target_flip_rate"][
        "one_sided_exchangeability_null"
    ]["n_m1"] = 2

    with pytest.raises(ValueError, match="group sizes do not match"):
        audit_reporting.prepare_true_flip_ablation_data(summary, _selection())
