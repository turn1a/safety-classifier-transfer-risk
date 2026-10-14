"""Aggregate local classifier predictions for every saved returned candidate.

The original attack search is unchanged. Source-success-gated transfer and applying
all returned candidates are separate retrospective policies with shared trial counts.
Neither policy supplies semantic or application-effectiveness evidence.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

_STATUS_COUNTS = (
    "source_successful_count",
    "source_failed_count",
    "source_skipped_count",
    "source_other_count",
)
_COUNT_COLUMNS = (
    "attempted",
    "source_successful",
    "source_failed",
    "source_skipped",
    "source_other",
    "target_original_positive",
    "target_original_benign",
    "target_original_positive_successful",
    "verified_flips_source_successful",
    "verified_flips_source_failed",
    "verified_flips_source_skipped",
    "verified_flips_source_other",
    "verified_flips_all_candidates",
    "originally_benign_candidates_benign",
)


def _validate_predictions(predictions: pd.DataFrame, labels: set[int]) -> None:
    """Require one binary outcome per pair and a consistent baseline per original."""
    required = {"pair_id", "original_id", "original_pred", "perturbed_pred"}
    if not required.issubset(predictions.columns) or predictions.empty:
        msg = "pair predictions must contain nonempty identifiers and binary labels"
        raise ValueError(msg)
    if predictions["pair_id"].duplicated().any() or predictions[list(required)].isna().any().any():
        msg = "pair predictions must be complete and unique"
        raise ValueError(msg)
    for column in ("original_pred", "perturbed_pred"):
        if not predictions[column].isin(labels).all():
            msg = f"{column} contains labels outside the target's binary label map"
            raise ValueError(msg)
    if (predictions.groupby("original_id")["original_pred"].nunique() != 1).any():
        msg = "target baseline predictions disagree for the same original"
        raise ValueError(msg)


def _validated_linkage(linkage: pd.DataFrame) -> pd.DataFrame:
    """Check integer multiplicities and exact source-status accounting."""
    identifiers = {"pair_id", "original_id", "source", "surrogate", "recipe"}
    counts = ["attempt_count", *_STATUS_COUNTS]
    if not identifiers.union(counts).issubset(linkage.columns) or linkage.empty:
        msg = "trial linkage must contain identifiers and all source-status counts"
        raise ValueError(msg)
    if linkage[list(identifiers)].isna().any().any():
        msg = "trial linkage contains missing identifiers"
        raise ValueError(msg)
    result = linkage.copy()
    for column in counts:
        values = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            msg = f"{column} must contain finite nonnegative integer counts"
            raise ValueError(msg)
        result[column] = values.astype("int64")
    if (result["attempt_count"] <= 0).any() or not (
        result[list(_STATUS_COUNTS)].sum(axis=1) == result["attempt_count"]
    ).all():
        msg = "source-status counts must sum to each positive trial multiplicity"
        raise ValueError(msg)
    if result.duplicated(["pair_id", "source", "surrogate", "recipe"]).any():
        msg = "trial linkage must aggregate duplicate pair/cell/source memberships"
        raise ValueError(msg)
    return result


def _annotated_counts(joined: pd.DataFrame, positive_label: int, benign_label: int) -> pd.DataFrame:
    """Convert pair outcomes into multiplicity-preserving trial count columns."""
    result = joined.copy()
    positive = result["original_pred"] == positive_label
    benign = result["original_pred"] == benign_label
    perturbed_benign = result["perturbed_pred"] == benign_label
    flip = positive & perturbed_benign
    result["attempted"] = result["attempt_count"]
    for status in ("successful", "failed", "skipped", "other"):
        result[f"source_{status}"] = result[f"source_{status}_count"]
        result[f"verified_flips_source_{status}"] = flip * result[f"source_{status}_count"]
    if result["verified_flips_source_skipped"].sum() != 0:
        msg = "a skipped source trial cannot contain a verified target prediction change"
        raise ValueError(msg)
    result["target_original_positive"] = positive * result["attempt_count"]
    result["target_original_benign"] = benign * result["attempt_count"]
    result["target_original_positive_successful"] = positive * result["source_successful_count"]
    result["verified_flips_all_candidates"] = flip * result["attempt_count"]
    result["originally_benign_candidates_benign"] = (benign & perturbed_benign) * result[
        "attempt_count"
    ]
    return result


def _aggregate(rows: pd.DataFrame, keys: list[str], target_id: str) -> pd.DataFrame:
    """Reduce to public count/rate columns without private identifiers or text."""
    result = rows.groupby(keys, as_index=False, sort=True)[list(_COUNT_COLUMNS)].sum()
    result.insert(0, "target", target_id)
    result["source_gated_yield"] = result["verified_flips_source_successful"] / result["attempted"]
    result["all_candidate_yield"] = result["verified_flips_all_candidates"] / result["attempted"]
    result["source_conditional_flip_rate"] = result["verified_flips_source_successful"] / result[
        "target_original_positive_successful"
    ].replace(0, np.nan)
    result["all_candidate_conditional_flip_rate"] = result[
        "verified_flips_all_candidates"
    ] / result["target_original_positive"].replace(0, np.nan)
    return result


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a finite rate or an explicit missing denominator."""
    return numerator / denominator if denominator else None


def build_retrospective_cells(
    predictions: pd.DataFrame,
    linkage: pd.DataFrame,
    *,
    target_id: str,
    positive_label: int = 1,
    benign_label: int = 0,
    excluded_surrogates: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Summarize both saved-candidate policies under an explicit target label map.

    Args:
        predictions: One private prediction row per exact original/perturbed pair.
        linkage: Private pair/cell/source memberships with all attempt multiplicities.
        target_id: Public target identifier.
        positive_label: Target's native injection or malicious class.
        benign_label: Target's native benign class.
        excluded_surrogates: Exact target self-source or defective historical source identifiers.

    Returns:
        Aggregate cells, aggregate source rows, and a non-text context report.

    Raises:
        ValueError: If labels, trial counts, baselines, or pair coverage are inconsistent.
    """
    if positive_label == benign_label or not target_id.strip():
        msg = "a named target and distinct positive/benign labels are required"
        raise ValueError(msg)
    _validate_predictions(predictions, {positive_label, benign_label})
    linked = _validated_linkage(linkage)
    linked = linked.loc[~linked["surrogate"].isin(excluded_surrogates)]
    if linked.empty:
        msg = "no trial records remain after target/source exclusions"
        raise ValueError(msg)
    joined = linked.merge(
        predictions,
        how="left",
        on=["pair_id", "original_id"],
        validate="many_to_one",
    )
    if joined[["original_pred", "perturbed_pred"]].isna().any().any():
        msg = "target predictions are missing for retained trial pairs"
        raise ValueError(msg)
    counts = _annotated_counts(joined, positive_label, benign_label)
    cells = _aggregate(counts, ["surrogate", "recipe"], target_id)
    sources = _aggregate(counts, ["source"], target_id)
    totals = {column: int(counts[column].sum()) for column in _COUNT_COLUMNS}
    originals = joined.drop_duplicates("original_id")
    context: dict[str, Any] = {
        "target": target_id,
        "excluded_surrogates": sorted(set(excluded_surrogates)),
        "included_surrogates": sorted(cells["surrogate"].unique().tolist()),
        "unique_pairs": int(joined["pair_id"].nunique()),
        "unique_originals": int(originals.shape[0]),
        "unique_original_positive": int((originals["original_pred"] == positive_label).sum()),
        "unique_original_benign": int((originals["original_pred"] == benign_label).sum()),
        "aggregate_counts": totals,
        "source_gated_yield": _rate(
            totals["verified_flips_source_successful"], totals["attempted"]
        ),
        "all_candidate_yield": _rate(totals["verified_flips_all_candidates"], totals["attempted"]),
        "source_conditional_flip_rate": _rate(
            totals["verified_flips_source_successful"],
            totals["target_original_positive_successful"],
        ),
        "all_candidate_conditional_flip_rate": _rate(
            totals["verified_flips_all_candidates"], totals["target_original_positive"]
        ),
        "analysis_status": "post_hoc_saved_candidate_policies",
        "unit_of_count": "repeated_trial_records",
        "semantic_validity": "not_assessed",
        "application_effectiveness": "not_assessed",
        "independent_selection_validation": False,
    }
    return cells, sources, context


def verify_reference_counts(
    current: pd.DataFrame,
    historical: pd.DataFrame,
    manifest: dict[str, Any],
    definition: dict[str, Any],
) -> dict[str, Any]:
    """Reject drift in source-success aggregates before interpreting the added candidate policy."""
    keys = ["surrogate", "recipe"]
    mapping = {
        "source_successful": "source_successful",
        "target_original_positive_successful": "target_original_injection",
        "verified_flips_source_successful": "true_target_flips",
    }
    prior = historical.loc[~historical["surrogate"].isin(definition["excluded_surrogates"])].copy()
    expected_keys = set(prior[keys].itertuples(index=False, name=None))
    current_keys = set(current[keys].itertuples(index=False, name=None))
    expected_historical_cells = int(manifest["n_surrogates"]) * int(manifest["n_recipes"])
    if len(historical) != expected_historical_cells or current_keys != expected_keys:
        msg = "local audit does not contain the configured historical reference grid"
        raise ValueError(msg)
    budget = int(manifest["attack_attempts_per_cell"])
    if "attempted" not in current or not (current["attempted"] == budget).all():
        msg = "local audit does not preserve the historical attempted budget per cell"
        raise ValueError(msg)
    merged = current.merge(
        prior, on=keys, suffixes=("_current", "_historical"), validate="one_to_one"
    )
    if len(merged) != len(current) or len(prior) != len(current):
        msg = "historical reference audit lacks an exact retained cell grid"
        raise ValueError(msg)
    for present, previous in mapping.items():
        left = f"{present}_current" if present == previous else present
        right = f"{previous}_historical" if present == previous else previous
        if not (merged[left] == merged[right]).all():
            msg = f"local audit does not reproduce historical {previous} cell counts"
            raise ValueError(msg)
    return {
        "counts_agree": True,
        "reference_cells_sha256": reference_cells_digest(current),
        "checked_cells": len(current),
        "checked_count_columns": list(mapping.values()),
        "retained_surrogates": sorted(prior["surrogate"].unique().tolist()),
        "aggregate_counts": {
            "attempted": int(current["attempted"].sum()),
            **{column: int(current[column].sum()) for column in mapping},
        },
        "interpretation": (
            "Aggregate prediction agreement does not establish historical weight identity."
        ),
    }


def reference_cells_digest(cells: pd.DataFrame) -> str:
    """Bind a reference receipt to the complete staged cell table."""
    serialized = (
        cells.sort_values(["surrogate", "recipe"])
        .sort_index(axis=1)
        .to_json(orient="records", double_precision=15)
    )
    return hashlib.sha256(serialized.encode()).hexdigest()
