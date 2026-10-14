"""Check current clean-positive misses against saved source eligibility counts."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd


def _source_counts(row: Mapping[str, Any]) -> tuple[int, int]:
    """Validate the canonical-positive and missed-positive counts for one source."""
    positive, missed = (
        row["canonical_positive_count"],
        row["canonical_positive_predicted_negative_count"],
    )
    if (
        any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in (positive, missed)
        )
        or missed > positive
    ):
        msg = "source baseline summary needs integer positive counts and bounded misses"
        raise ValueError(msg)
    return int(positive), int(missed)


def verify_source_baseline_counts(
    linkage: pd.DataFrame,
    baseline: Mapping[str, Any],
    *,
    recipes: Sequence[str],
    attempts_per_cell: int,
) -> dict[str, Any]:
    """Require model/source/recipe parity before interpreting current clean baselines.

    Args:
        linkage: Private saved source-status counts with no text required.
        baseline: Current detector clean-input aggregate report.
        recipes: Required recipe names.
        attempts_per_cell: Canonical-positive originals per model/recipe cell.

    Returns:
        Public count-check rows without private pair or original identifiers.

    Raises:
        ValueError: If models, grids, counts, or source-specific misses differ.
    """
    keys = ["surrogate", "source", "recipe"]
    counts = ["attempt_count", "source_skipped_count"]
    if not set(keys + counts).issubset(linkage.columns) or linkage[keys].isna().any().any():
        msg = "source baseline linkage lacks complete identifiers and count columns"
        raise ValueError(msg)
    reports = [report for report in baseline["reports"] if report["target_id"] != "protectai-v2"]
    models = sorted(report["target_id"] for report in reports)
    actual_models = set(linkage["surrogate"]) - {"bilstm-attention"}
    if len(models) != len(set(models)) or actual_models != set(models) or not models:
        msg = "source baseline model identities differ from the saved sweep"
        raise ValueError(msg)
    retained = linkage[linkage["surrogate"].isin(models)]
    values = retained[counts].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
        msg = "source baseline counts must be finite nonnegative integers"
        raise ValueError(msg)
    grouped = retained.groupby(keys)[counts].sum()
    expected: dict[tuple[str, str, str], tuple[int, int]] = {}
    for report in reports:
        model = report["target_id"]
        source_rows = report["by_source"]
        if len({row["source"] for row in source_rows}) != len(source_rows):
            msg = "source baseline summary repeats a source identity"
            raise ValueError(msg)
        if sum(row["canonical_positive_count"] for row in source_rows) != attempts_per_cell:
            msg = "source baseline summary has a different positive-input budget"
            raise ValueError(msg)
        for row in source_rows:
            positive, missed = _source_counts(row)
            if positive:
                for recipe in recipes:
                    expected[(model, row["source"], recipe)] = (positive, missed)
    if set(grouped.index) != set(expected):
        msg = "source baseline linkage lacks the complete model/source/recipe grid"
        raise ValueError(msg)
    checks = []
    observed_rows = grouped.to_dict(orient="index")
    for key, (attempted, missed) in sorted(expected.items()):
        observed = observed_rows[key]
        if observed["attempt_count"] != attempted or observed["source_skipped_count"] != missed:
            msg = "current clean-positive misses differ from saved source baseline counts"
            raise ValueError(msg)
        checks.append(
            {
                "surrogate": key[0],
                "source": key[1],
                "recipe": key[2],
                "attempted": attempted,
                "current_clean_positive_misses": missed,
                "saved_baseline_skips": int(observed["source_skipped_count"]),
            }
        )
    return {
        "counts_agree": True,
        "models_checked": models,
        "cells_checked": len(checks),
        "rows": checks,
        "interpretation": (
            "Current positive-input misses match saved skips by source and recipe. "
            "Count agreement does not prove historical checkpoint identity."
        ),
    }
