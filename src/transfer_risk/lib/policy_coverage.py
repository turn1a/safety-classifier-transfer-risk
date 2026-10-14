"""Measure per-original union coverage for saved equal-trial source selections."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

from transfer_risk.lib.retrospective_audit import build_retrospective_cells

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


def _subset_coverage(
    sources: Sequence[str],
    gated: dict[str, set[str]],
    all_candidates: dict[str, set[str]],
    positive_originals: int,
) -> dict[str, Any]:
    """Count each affected target-positive original once across a selected source set."""
    source_gated = len(set.union(*(gated[source] for source in sources)))
    candidate_count = len(set.union(*(all_candidates[source] for source in sources)))
    return {
        "surrogates": list(sources),
        "source_gated_originals": source_gated,
        "all_candidate_originals": candidate_count,
        "source_gated_positive_coverage": source_gated / positive_originals
        if positive_originals
        else None,
        "all_candidate_positive_coverage": candidate_count / positive_originals
        if positive_originals
        else None,
    }


def summarize_original_policy_coverage(
    predictions: pd.DataFrame,
    linkage: pd.DataFrame,
    similarity: pd.DataFrame,
    *,
    attempts_per_cell: int,
    recipes: Sequence[str],
    excluded_surrogates: Sequence[str] = ("bilstm-attention",),
    selection_size: int = 3,
) -> dict[str, Any]:
    """Enumerate saved source selections using distinct affected originals as the unit.

    Args:
        predictions: Private reference-target binary predictions for exact text pairs.
        linkage: Private pair-to-original/source/recipe attempt accounting.
        similarity: Complete historical source identities and mean CKA.
        attempts_per_cell: Complete evaluation-original count per source/recipe cell.
        recipes: Required historical recipe identifiers.
        excluded_surrogates: Sources omitted for the implementation sensitivity.
        selection_size: Equal source count for every enumerated selection.

    Returns:
        Aggregate coverage, observed subset comparisons, and no private input identifiers.

    Raises:
        ValueError: If predictions, trial grids, similarity identities, or subset size fail.
    """
    cells, _, _ = build_retrospective_cells(
        predictions, linkage, target_id="protectai-v2", excluded_surrogates=excluded_surrogates
    )
    models = sorted(set(similarity["surrogate"]) - set(excluded_surrogates))
    if similarity["surrogate"].duplicated().any() or not 1 <= selection_size <= len(models):
        msg = "original-coverage source identities or selection size are invalid"
        raise ValueError(msg)
    expected = {(model, recipe) for model in models for recipe in recipes}
    if (
        set(zip(cells["surrogate"], cells["recipe"], strict=True)) != expected
        or len(cells) != len(expected)
        or not (cells["attempted"] == attempts_per_cell).all()
    ):
        msg = "original-coverage cells lack the complete fixed-trial grid"
        raise ValueError(msg)
    retained = linkage[linkage["surrogate"].isin(models)]
    trial_counts = retained.groupby(["original_id", "surrogate", "recipe"])["attempt_count"].sum()
    originals = set(retained["original_id"])
    if (
        len(originals) != attempts_per_cell
        or len(trial_counts) != attempts_per_cell * len(expected)
        or not (trial_counts == 1).all()
    ):
        msg = "original-coverage grid must evaluate each original once per source/recipe"
        raise ValueError(msg)
    evidence = retained.merge(
        predictions[["pair_id", "original_pred", "perturbed_pred"]],
        on="pair_id",
        validate="many_to_one",
    )
    positive = int(evidence.drop_duplicates("original_id")["original_pred"].sum())
    flip = (evidence["original_pred"] == 1) & (evidence["perturbed_pred"] == 0)
    all_candidates = {
        model: set(evidence.loc[flip & (evidence["surrogate"] == model), "original_id"])
        for model in models
    }
    gated = {
        model: set(
            evidence.loc[
                flip & (evidence["surrogate"] == model) & (evidence["source_successful_count"] > 0),
                "original_id",
            ]
        )
        for model in models
    }
    cka = similarity.set_index("surrogate")["mean_cka"].to_dict()
    if not np.isfinite([cka[model] for model in models]).all():
        msg = "original-coverage CKA values must be finite"
        raise ValueError(msg)
    selected_models = sorted(sorted(models, key=lambda name: (-cka[name], name))[:selection_size])
    selected = _subset_coverage(selected_models, gated, all_candidates, positive)
    subsets = [
        _subset_coverage(subset, gated, all_candidates, positive)
        for subset in combinations(models, selection_size)
    ]
    comparisons = {}
    for policy in ("source_gated", "all_candidate"):
        key = f"{policy}_originals"
        values = [row[key] for row in subsets]
        comparisons[policy] = {
            "n_strictly_higher": sum(value > selected[key] for value in values),
            "n_equal": sum(value == selected[key] for value in values),
            "mean_originals": float(np.mean(values)),
            "best_originals": max(values),
            "best_selections": [row["surrogates"] for row in subsets if row[key] == max(values)],
        }
    return {
        "analysis_status": "post_hoc_saved_original_union_coverage",
        "originals": len(originals),
        "original_target_positive": positive,
        "surrogates": models,
        "selected_surrogates": selected_models,
        "attempted_trials_per_subset": selection_size * len(recipes) * attempts_per_cell,
        "n_subsets": len(subsets),
        "selected": selected,
        "whole_pool": _subset_coverage(models, gated, all_candidates, positive),
        "subsets": subsets,
        "comparisons": comparisons,
        "semantic_validity_assessed": False,
        "independent_selection_validation": False,
        "cka_tie_breaker": "ascending surrogate identifier",
        "interpretation": (
            "Each affected original is counted once per policy. All subsets use the same "
            "saved inputs and trial budgets; ranks are descriptive and query costs unknown."
        ),
    }
