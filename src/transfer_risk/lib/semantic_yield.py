"""Literal valid-flip bounds from private human consensus and saved target predictions.

The unit is the retained repeated trial record. A reviewed exact pair can contribute
to several trials. Unknown validity remains unknown; this module supplies no yield
point estimator, cell-rate estimates, original-union estimate, or confidence interval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from transfer_risk.lib.retrospective_audit import build_retrospective_cells

if TYPE_CHECKING:
    from collections.abc import Sequence


def _sample_statuses(
    consensus: pd.DataFrame, linkage: pd.DataFrame
) -> tuple[dict[str, str], set[str]]:
    """Validate consensus against the complete saved sample before source exclusions."""
    if "selected_for_review" not in linkage or not all(
        isinstance(value, (bool, np.bool_)) for value in linkage["selected_for_review"]
    ):
        msg = "linkage selected_for_review must contain explicit boolean values"
        raise ValueError(msg)
    if (linkage.groupby("pair_id")["selected_for_review"].nunique() != 1).any():
        msg = "sample selection must agree across all memberships of each pair"
        raise ValueError(msg)
    selected = set(linkage.loc[linkage["selected_for_review"], "pair_id"])
    if (
        not {"pair_id", "status"}.issubset(consensus.columns)
        or consensus["pair_id"].duplicated().any()
        or set(consensus["pair_id"]) != selected
    ):
        msg = "consensus must cover exactly the globally selected pair sample"
        raise ValueError(msg)
    if not consensus["status"].isin(["valid", "invalid", "unresolved"]).all():
        msg = "consensus status must be valid, invalid or unresolved"
        raise ValueError(msg)
    return dict(zip(consensus["pair_id"], consensus["status"], strict=True)), selected


def _ratio(numerator: int, denominator: int) -> float | None:
    """Return a literal finite-population fraction or an absent denominator."""
    return numerator / denominator if denominator else None


def _policy_bounds(
    masses: pd.Series[Any], statuses: pd.Series[Any], selected: pd.Series[Any], attempted: int
) -> dict[str, Any]:
    """Aggregate known validity and unknown remainder without extrapolating sample labels."""
    contributing = masses > 0
    resolved = statuses.isin(["valid", "invalid"])
    total = int(masses.sum())
    valid = int(masses.loc[statuses == "valid"].sum())
    invalid = int(masses.loc[statuses == "invalid"].sum())
    unknown = total - valid - invalid
    sampled_mass = int(masses.loc[selected].sum())
    resolved_mass = valid + invalid
    return {
        "attempted_trials": attempted,
        "classifier_flip_trials": total,
        "confirmed_valid_flip_trials": valid,
        "confirmed_invalid_flip_trials": invalid,
        "unresolved_flip_trials": unknown,
        "valid_flip_count_bounds": {"lower": valid, "upper": valid + unknown},
        "valid_flip_yield_bounds": {
            "lower": _ratio(valid, attempted),
            "upper": _ratio(valid + unknown, attempted),
        },
        "classifier_flip_pairs": int(contributing.sum()),
        "sampled_classifier_flip_pairs": int((contributing & selected).sum()),
        "resolved_classifier_flip_pairs": int((contributing & resolved).sum()),
        "sampled_classifier_flip_trials": sampled_mass,
        "resolved_classifier_flip_trials": resolved_mass,
        "sampled_classifier_flip_fraction": _ratio(sampled_mass, total),
        "resolved_classifier_flip_fraction": _ratio(resolved_mass, total),
        "estimate": None,
    }


def summarize_valid_flip_yield(
    consensus: pd.DataFrame,
    predictions: pd.DataFrame,
    linkage: pd.DataFrame,
    *,
    target_id: str,
    positive_label: int = 1,
    benign_label: int = 0,
    excluded_surrogates: Sequence[str] = (),
) -> dict[str, Any]:
    """Bound valid target prediction-change yields for two saved-candidate policies.

    Args:
        consensus: Validated private human consensus for every globally sampled pair.
        predictions: Saved private target labels on exact originals and perturbations.
        linkage: Full private pair/cell/source multiplicities and sample-selection flags.
        target_id: Public target identifier.
        positive_label: Native target injection or malicious class.
        benign_label: Native target benign class.
        excluded_surrogates: Historical sources excluded from this target's retained pool.

    Returns:
        Aggregate-only source-success-gated and all-returned-candidate bounds. Each
        lower count includes confirmed-valid classifier flips; its upper count also
        includes every unresolved flip. Yield divides both by all retained attempted
        trials, including originally benign and unsuccessful-source trials. Resolution
        coverage and full-review completeness are different quantities. No sample labels
        are extrapolated, and no valid-flip yield point estimate is supplied.

    Raises:
        ValueError: If target predictions, trial accounting, sample membership, selection
            flags, or consensus statuses are inconsistent.
    """
    _, _, context = build_retrospective_cells(
        predictions,
        linkage,
        target_id=target_id,
        positive_label=positive_label,
        benign_label=benign_label,
        excluded_surrogates=excluded_surrogates,
    )
    statuses, selected_ids = _sample_statuses(consensus, linkage)
    retained = linkage.loc[~linkage["surrogate"].isin(excluded_surrogates)].copy()
    for column in ("attempt_count", "source_successful_count"):
        retained[column] = pd.to_numeric(retained[column], errors="raise").astype("int64")
    joined = retained.merge(
        predictions[["pair_id", "original_id", "original_pred", "perturbed_pred"]],
        on=["pair_id", "original_id"],
        how="left",
        validate="many_to_one",
    )
    flip = (joined["original_pred"] == positive_label) & (joined["perturbed_pred"] == benign_label)
    joined["all_flip_mass"] = joined["attempt_count"] * flip
    joined["gated_flip_mass"] = joined["source_successful_count"] * flip
    pairs = joined.groupby("pair_id")[["all_flip_mass", "gated_flip_mass"]].sum()
    pair_status = pairs.index.to_series().map(statuses).fillna("unresolved")
    pair_selected = pairs.index.to_series().isin(selected_ids)
    attempted = int(context["aggregate_counts"]["attempted"])
    resolved_selected = sum(status != "unresolved" for status in statuses.values())
    return {
        "schema_version": "semantic-valid-flip-yield-v1",
        "target": target_id,
        "target_label_map": {"positive": positive_label, "benign": benign_label},
        "included_surrogates": context["included_surrogates"],
        "excluded_surrogates": sorted(set(excluded_surrogates)),
        "unit_of_count": "retained repeated trial records",
        "attempted_trials": attempted,
        "selected_pairs": len(selected_ids),
        "resolved_selected_pairs": resolved_selected,
        "selected_judgments_resolved": bool(selected_ids)
        and resolved_selected == len(selected_ids),
        "policies": {
            "source_success_gated": _policy_bounds(
                pairs["gated_flip_mass"], pair_status, pair_selected, attempted
            ),
            "all_returned_candidates": _policy_bounds(
                pairs["all_flip_mass"], pair_status, pair_selected, attempted
            ),
        },
        "bounds_are_confidence_intervals": False,
        "cell_rates_estimated": False,
        "cka_validity_association": None,
        "original_union_estimated": False,
        "point_estimate_status": (
            "withheld; no trial-yield survey estimator or uncertainty analysis implemented"
        ),
        "interpretation": (
            "Both policies use all retained attempts as the yield denominator. Source-success "
            "gating permits only saved successful-source candidates; the all-returned policy "
            "also permits failed or skipped returned candidates. Unknown validity contributes "
            "only to the upper bound. Bounds use observed consensus without probability weighting."
        ),
        "limitations": [
            "A sparse pair sample does not establish semantic validity of original-union coverage.",
            (
                "Pair-validity Hajek estimation is reported separately and is not a "
                "valid-flip yield estimate."
            ),
            (
                "Semantic validity of a classifier prediction change does not establish "
                "application effectiveness."
            ),
        ],
    }
