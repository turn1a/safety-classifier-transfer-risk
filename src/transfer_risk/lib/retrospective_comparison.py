"""Compare saved-candidate policies with fixed source pools and explicit baselines."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np

from transfer_risk.lib.association import spearman_association
from transfer_risk.lib.retrospective_audit import reference_cells_digest

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
    "originally_benign_candidates_benign",
)
_MIN_MODELS = 2
_INFERENCE_FIELDS = (
    "inference_weights_sha256",
    "tokenizer_vocab_sha256",
    "loaded_hub_revision",
    "inference_dtype",
    "max_seq_len",
    "device",
    "provenance_scope",
)


def _validate_inference_values(record: Mapping[str, Any], *, target: str) -> None:
    """Require usable saved identities and typed inference settings before comparing them."""
    for field in ("inference_weights_sha256", "tokenizer_vocab_sha256"):
        value = record.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            msg = f"independent inference context lacks a valid SHA-256 for {target}: {field}"
            raise ValueError(msg)
    for field in ("inference_dtype", "device", "provenance_scope"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            msg = f"independent inference context requires a nonempty string for {target}: {field}"
            raise ValueError(msg)
    revision = record.get("loaded_hub_revision")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        msg = f"independent inference context has an invalid loaded_hub_revision for {target}"
        raise ValueError(msg)
    limit = record.get("max_seq_len")
    if type(limit) is not int or limit <= 0:
        msg = f"independent inference context requires a positive integer max_seq_len for {target}"
        raise ValueError(msg)


def _validate_inference_context(
    context: Mapping[str, Any], baseline_context: Mapping[str, Any]
) -> None:
    """Require matching independently saved clean-input model identities and settings."""
    reports = baseline_context.get("reports", [])
    baselines = {report["target_id"]: report for report in reports}
    if len(baselines) != len(reports):
        msg = "independent baseline context contains duplicate target identities"
        raise ValueError(msg)
    for report in context["reports"]:
        target = report["target"]
        baseline = baselines.get(target)
        if baseline is None:
            msg = f"independent inference context is missing target {target}"
            raise ValueError(msg)
        _validate_inference_values(report, target=target)
        _validate_inference_values(baseline, target=target)
        for field in _INFERENCE_FIELDS:
            if field not in report or field not in baseline or report[field] != baseline[field]:
                msg = f"independent inference context differs for {target}: {field}"
                raise ValueError(msg)
        labels = baseline.get("label_values", {})
        label_values = (
            report.get("positive_label"),
            report.get("benign_label"),
            labels.get("canonical_positive"),
            labels.get("canonical_negative"),
        )
        if (
            any(type(value) is not int or value < 0 for value in label_values)
            or report["positive_label"] != labels.get("canonical_positive")
            or report["benign_label"] != labels.get("canonical_negative")
            or report["positive_label"] == report["benign_label"]
        ):
            msg = f"independent inference context differs for {target}: label mapping"
            raise ValueError(msg)
        pooled = baseline.get("pooled", {})
        fields = {
            "unique_originals": "canonical_positive_count",
            "unique_original_positive": "canonical_positive_predicted_positive_count",
            "unique_original_benign": "canonical_positive_predicted_negative_count",
        }
        if any(report[name] != pooled.get(field) for name, field in fields.items()):
            msg = f"independent clean-input baseline counts differ for {target}"
            raise ValueError(msg)


def _validate_context(
    cells_by_target: Mapping[str, pd.DataFrame],
    context: Mapping[str, Any],
    reference_target: str,
    attempts: int,
) -> dict[str, list[str]]:
    """Reconcile current tables with the reference receipt and every declared total."""
    receipt = context.get("reference_check", {})
    if receipt.get("counts_agree") is not True or reference_target not in cells_by_target:
        msg = "saved-candidate comparison requires a successful historical reference check"
        raise ValueError(msg)
    reports = context.get("reports", [])
    declared = {report["target"]: report["included_surrogates"] for report in reports}
    if len(declared) != len(reports) or set(declared) != set(cells_by_target):
        msg = "saved-candidate comparison target identities differ from the inference context"
        raise ValueError(msg)
    reference = cells_by_target[reference_target]
    if receipt.get("reference_cells_sha256") != reference_cells_digest(reference):
        msg = "saved-candidate comparison rejected a stale reference receipt"
        raise ValueError(msg)
    if receipt.get("retained_surrogates") != sorted(reference["surrogate"].unique()) or receipt.get(
        "checked_cells"
    ) != len(reference):
        msg = "reference receipt differs from the retained source pool or cell count"
        raise ValueError(msg)
    checked_counts = (
        "attempted",
        "source_successful",
        "target_original_positive_successful",
        "verified_flips_source_successful",
    )
    if receipt.get("aggregate_counts") != {
        column: int(reference[column].sum()) for column in checked_counts
    }:
        msg = "reference receipt totals differ from the current cells"
        raise ValueError(msg)
    pools = []
    for report in reports:
        target = report["target"]
        included, excluded = report["included_surrogates"], report.get("excluded_surrogates", [])
        if (
            included != sorted(set(included))
            or excluded != sorted(set(excluded))
            or set(included) & set(excluded)
            or (target != reference_target and target in included)
        ):
            msg = "retrospective context has inconsistent retained or excluded source pools"
            raise ValueError(msg)
        pools.append(set(included) | set(excluded))
        cells = cells_by_target[target]
        totals = {column: int(cells[column].sum()) for column in _COUNTS}
        if report.get("aggregate_counts") != totals:
            msg = f"retrospective context totals differ from current cells for {target}"
            raise ValueError(msg)
        if (
            report.get("unique_originals") != attempts
            or report.get("unique_original_positive") != cells["target_original_positive"].iloc[0]
            or report.get("unique_original_benign") != cells["target_original_benign"].iloc[0]
        ):
            msg = f"retrospective original counts differ from current cells for {target}"
            raise ValueError(msg)
    if any(pool != pools[0] for pool in pools):
        msg = "retrospective contexts declare different full source pools"
        raise ValueError(msg)
    return declared


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
    baseline_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Reduce common-source target checks and the reference all-candidate CKA association.

    Args:
        cells_by_target: Aggregate cells from the completed local target pass.
        similarity: Historical surrogate CKA summaries.
        context: Completed reference-check receipt and declared retained pools.
        attempts_per_cell: Saved fixed attempted budget.
        recipes: Complete configured recipe grid.
        reference_target: Historical target identity.
        baseline_context: Independent public clean-input identity and settings evidence.

    Returns:
        Aggregate-only common-pool comparison and reference policy sensitivity.

    Raises:
        ValueError: If receipts, grids, counts, or model identities are inconsistent.
    """
    # Validate the table shape before context reconciliation indexes required columns.
    declared = {report["target"]: report["included_surrogates"] for report in context["reports"]}
    if set(declared) != set(cells_by_target):
        msg = "saved-candidate comparison targets differ from the inference context"
        raise ValueError(msg)
    for target, cells in cells_by_target.items():
        _validate_cells(cells, target, declared[target], attempts_per_cell, recipes)
    declared = _validate_context(cells_by_target, context, reference_target, attempts_per_cell)
    _validate_inference_context(context, baseline_context)
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
        else {
            "rho": None,
            "two_sided_p": None,
            "n": len(source_rows),
            "exact": False,
            "status": "not_estimable",
            "reason": "Spearman association is undefined for constant input sequences",
        }
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
        "inference_context_check": {
            "identity_and_settings_agree": True,
            "checked_fields": [*_INFERENCE_FIELDS, "positive_label", "benign_label"],
            "scope": (
                "Current saved candidate and clean-input contexts; "
                "historical weights are unverified."
            ),
        },
        "limitations": [
            "Returned candidates from failed source searches define a separate post-hoc policy.",
            "Different target objectives and operating points prevent a robustness ranking.",
            "All checkpoints share the DeBERTa family; no independent CKA selector was calibrated.",
            "Prediction changes have unresolved semantic validity and application effectiveness.",
            "Trial budgets are fixed; historical per-trial query costs were not saved.",
        ],
    }
