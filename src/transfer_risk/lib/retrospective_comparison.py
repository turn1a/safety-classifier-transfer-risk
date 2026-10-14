"""Compare saved-candidate policies with fixed source pools and explicit baselines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from transfer_risk.lib.association import spearman_association

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

_STATUSES = ("successful", "failed", "skipped", "other")
_COUNTS = (
    "attempted",
    *(f"source_{status}" for status in _STATUSES),
    "target_original_positive",
    "target_original_benign",
    "target_original_positive_successful",
    *(f"verified_flips_source_{status}" for status in _STATUSES),
    "verified_flips_all_candidates",
)
_MIN_MODELS = 2


def _validate_cells(
    cells: pd.DataFrame,
    target: str,
    models: Sequence[str],
    attempts: int,
    recipes: Sequence[str],
) -> None:
    """Require the declared full grid and consistent count identities before reduction."""
    required = {"target", "surrogate", "recipe", *_COUNTS}
    if not required.issubset(cells.columns) or cells.columns.duplicated().any():
        msg = "retrospective policy cells lack required unique columns"
        raise ValueError(msg)
    expected = {(model, recipe) for model in models for recipe in recipes}
    actual = set(zip(cells["surrogate"], cells["recipe"], strict=True))
    if actual != expected or len(cells) != len(expected) or set(cells["target"]) != {target}:
        msg = "retrospective policy cells do not contain the declared target/source/recipe grid"
        raise ValueError(msg)
    values = cells[list(_COUNTS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
        msg = "retrospective policy counts must be finite nonnegative integers"
        raise ValueError(msg)
    source_total = cells[[f"source_{status}" for status in _STATUSES]].sum(axis=1)
    flip_total = cells[[f"verified_flips_source_{status}" for status in _STATUSES]].sum(axis=1)
    if (
        not (cells["attempted"] == attempts).all()
        or not (source_total == cells["attempted"]).all()
        or not (flip_total == cells["verified_flips_all_candidates"]).all()
        or not (
            (cells["target_original_positive"] + cells["target_original_benign"]) == attempts
        ).all()
        or cells["target_original_positive"].nunique() != 1
        or (cells["verified_flips_all_candidates"] > cells["target_original_positive"]).any()
        or (
            cells["verified_flips_source_successful"] > cells["target_original_positive_successful"]
        ).any()
        or (cells["target_original_positive_successful"] > cells["source_successful"]).any()
    ):
        msg = "retrospective policy count identities or attempted budgets are inconsistent"
        raise ValueError(msg)
    for status in _STATUSES:
        if (cells[f"verified_flips_source_{status}"] > cells[f"source_{status}"]).any():
            msg = "retrospective flips exceed their source-status cohort"
            raise ValueError(msg)


def _pool_row(cells: pd.DataFrame, target: str) -> dict[str, Any]:
    """Reduce one fixed source pool while retaining repeated-trial and unique baselines."""
    attempted = int(cells["attempted"].sum())
    gated = int(cells["verified_flips_source_successful"].sum())
    all_candidates = int(cells["verified_flips_all_candidates"].sum())
    original_positive = int(cells["target_original_positive"].iloc[0])
    return {
        "target": target,
        "n_surrogates": int(cells["surrogate"].nunique()),
        "attempted": attempted,
        "original_target_positive": original_positive,
        "original_target_benign": int(cells["target_original_benign"].iloc[0]),
        "source_gated_flips": gated,
        "failed_source_flips": int(cells["verified_flips_source_failed"].sum()),
        "all_candidate_flips": all_candidates,
        "source_gated_yield": gated / attempted,
        "all_candidate_yield": all_candidates / attempted,
        "all_candidate_target_positive_rate": (
            all_candidates / int(cells["target_original_positive"].sum())
            if original_positive
            else None
        ),
    }


def compare_saved_candidate_policies(
    cells_by_target: Mapping[str, pd.DataFrame],
    similarity: pd.DataFrame,
    context: Mapping[str, Any],
    *,
    attempts_per_cell: int,
    recipes: Sequence[str],
    reference_target: str,
) -> dict[str, Any]:
    """Reduce common-source target checks and the reference all-candidate CKA association.

    Args:
        cells_by_target: Aggregate cells from the completed local target pass.
        similarity: Historical surrogate CKA summaries.
        context: Completed reference-check receipt and declared retained pools.
        attempts_per_cell: Saved fixed attempted budget.
        recipes: Complete configured recipe grid.
        reference_target: Historical target identity.

    Returns:
        Aggregate-only common-pool comparison and reference policy sensitivity.

    Raises:
        ValueError: If receipts, grids, counts, or model identities are inconsistent.
    """
    if context.get("reference_check", {}).get("counts_agree") is not True:
        msg = "saved-candidate comparison requires a successful historical reference check"
        raise ValueError(msg)
    declared = {report["target"]: report["included_surrogates"] for report in context["reports"]}
    if set(declared) != set(cells_by_target) or reference_target not in declared:
        msg = "saved-candidate comparison target identities differ from the inference context"
        raise ValueError(msg)
    for target, cells in cells_by_target.items():
        _validate_cells(cells, target, declared[target], attempts_per_cell, recipes)
    common = sorted(set.intersection(*(set(models) for models in declared.values())))
    if len(common) < _MIN_MODELS:
        msg = "saved-candidate comparison requires at least two common sources"
        raise ValueError(msg)
    if similarity["surrogate"].duplicated().any():
        msg = "historical similarity identities must be unique"
        raise ValueError(msg)
    cka = similarity.set_index("surrogate")["mean_cka"].to_dict()
    reference = cells_by_target[reference_target]
    source_rows = []
    for model in sorted(declared[reference_target]):
        if model not in cka:
            msg = "saved-candidate source is missing from historical CKA summaries"
            raise ValueError(msg)
        row = _pool_row(reference[reference["surrogate"] == model], reference_target)
        source_rows.append({"surrogate": model, "mean_cka": float(cka[model]), **row})
    yields = [row["all_candidate_yield"] for row in source_rows]
    association = (
        spearman_association([row["mean_cka"] for row in source_rows], yields)
        if len(set(yields)) > 1 and len({row["mean_cka"] for row in source_rows}) > 1
        else None
    )
    return {
        "analysis_status": "post_hoc_saved_candidate_policy_sensitivity",
        "common_surrogates": common,
        "n_recipes": len(recipes),
        "common_pool": [
            _pool_row(cells[cells["surrogate"].isin(common)], target)
            for target, cells in cells_by_target.items()
        ],
        "reference_pool": {
            "target": reference_target,
            **_pool_row(reference, reference_target),
            "surrogate_rows": source_rows,
            "association": association,
            "outcome": "all_returned_candidate_target_positive_to_benign_yield",
        },
        "independent_selection_validation": False,
        "limitations": [
            "Returned candidates from failed source searches define a separate post-hoc policy.",
            "Different target objectives and operating points prevent a robustness ranking.",
            "All checkpoints share the DeBERTa family; no independent CKA selector was calibrated.",
            "Prediction changes have unresolved semantic validity and application effectiveness.",
            "Trial budgets are fixed; historical per-trial query costs were not saved.",
        ],
    }
