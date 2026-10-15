"""Frozen trial sampling and finite-population human-validity reductions without I/O."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from transfer_risk.lib.association import spearman_association
from transfer_risk.lib.retrospective_audit import build_retrospective_cells
from transfer_risk.lib.seeds import derive_seeds
from transfer_risk.lib.semantic_review_tool import build_staged_review_dataset

if TYPE_CHECKING:
    from collections.abc import Mapping

_SCHEMA = "semantic-flip-cohort-v1"
_KEYS = ["pair_id", "original_id", "source", "surrogate", "recipe"]
_STATUSES = ["successful", "failed", "skipped", "other"]
_TAIL_PROBABILITY = 0.025
_MIN_ASSOCIATION_UNITS = 2


def fingerprint(value: Any) -> str:
    """Hash deterministic exact JSON, retaining Unicode and rejecting nonfinite values."""
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_flip_cohort(
    pairs: pd.DataFrame,
    linkage: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    seed: int,
    sample_size: int = 300,
) -> dict[str, Any]:
    """Freeze an SRS of changed trial occurrences before deduplicating reviewer display.

    Args:
        pairs: Existing exact private pair texts and original identities.
        linkage: Existing all-outcome repeated-trial counts.
        predictions: Saved ProtectAI reference predictions; no inference is performed.
        seed: Project root seed, derived through the project NumPy seed stream.
        sample_size: Fixed trial sample size, capped at the changed population size.

    Returns:
        Private versioned cohort with trial identities, frozen draw, blinded queue,
        original universe and complete attempted-trial cell denominators.

    Raises:
        ValueError: If saved pair coverage, exact identities or sample size are invalid.
    """
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        msg = "sample_size must be a positive integer"
        raise ValueError(msg)
    cells, _, context = build_retrospective_cells(
        predictions, linkage, target_id="protectai-v2", excluded_surrogates=["bilstm-attention"]
    )
    if pairs["pair_id"].duplicated().any():
        msg = "private pairs must be unique"
        raise ValueError(msg)
    retained = linkage.loc[linkage["surrogate"] != "bilstm-attention"].copy()
    joined = retained.merge(predictions, on=["pair_id", "original_id"], validate="many_to_one")
    text = joined.merge(
        pairs[["pair_id", "original_id", "original", "perturbed"]],
        on=["pair_id", "original_id"],
        how="left",
        validate="many_to_one",
    )
    if text[["original", "perturbed"]].isna().any().any():
        msg = "private exact pairs are missing from saved prediction linkage"
        raise ValueError(msg)
    flipped = text.loc[(text["original_pred"] == 1) & (text["perturbed_pred"] == 0)]
    trials: list[dict[str, Any]] = []
    for row in cast("list[dict[str, Any]]", flipped.sort_values(_KEYS).to_dict("records")):
        membership = {key: str(row[key]) for key in _KEYS}
        for status in _STATUSES:
            for index in range(int(row[f"source_{status}_count"])):
                trials.append(
                    {
                        **membership,
                        "source_success": status == "successful",
                        "occurrence_id": fingerprint([membership, status, index]),
                    }
                )
    trials.sort(key=lambda row: row["occurrence_id"])
    rng = np.random.default_rng(derive_seeds(seed).numpy)
    n = min(sample_size, len(trials))
    draw = [trials[int(i)]["occurrence_id"] for i in rng.choice(len(trials), size=n, replace=False)]
    trial_by_id = {row["occurrence_id"]: row for row in trials}
    stage_one = list(dict.fromkeys(trial_by_id[identity]["pair_id"] for identity in draw))
    remaining = sorted(set(flipped["pair_id"]) - set(stage_one))
    queue = stage_one + [remaining[int(i)] for i in rng.permutation(len(remaining))]
    texts = flipped.drop_duplicates("pair_id").set_index("pair_id")
    candidates = [
        {
            "pair_id": identity,
            "original": str(texts.loc[identity, "original"]),
            "perturbed": str(texts.loc[identity, "perturbed"]),
        }
        for identity in queue
    ]
    dataset = (
        build_staged_review_dataset(
            candidates,
            stage_one_pair_count=len(stage_one),
            sample_trial_count=n,
            cohort_binding=fingerprint(
                {
                    "trials": trials,
                    "sample": draw,
                    "denominators": cells[["surrogate", "recipe", "attempted"]].to_dict("records"),
                }
            ),
        )
        if candidates
        else None
    )
    value: dict[str, Any] = {
        "schema_version": _SCHEMA,
        "attempted_trials": int(context["aggregate_counts"]["attempted"]),
        "original_ids": sorted(set(retained["original_id"])),
        "target_original_positive_originals": context["unique_original_positive"],
        "cells": cells[["surrogate", "recipe", "attempted"]]
        .sort_values(["surrogate", "recipe"])
        .to_dict("records"),
        "trials": trials,
        "sample_occurrence_ids": draw,
        "review_dataset": dataset,
        "sampling": {
            "root_seed": seed,
            "derived_numpy_seed": derive_seeds(seed).numpy,
            "requested_sample_trials": sample_size,
            "sample_trials": n,
            "procedure": (
                "NumPy Generator.choice without replacement on sorted exact occurrence identities; "
                "first-occurrence pair queue, then independently permuted remaining pair identities"
            ),
            "frame": "all retained target injection-to-benign trial occurrences",
        },
        "saved_input_fingerprints": {
            name: fingerprint(frame.sort_values(keys).to_dict("records"))
            for name, frame, keys in [
                ("pairs", pairs, ["pair_id"]),
                ("linkage", linkage, _KEYS),
                ("predictions", predictions, ["pair_id"]),
            ]
        },
    }
    return {**value, "cohort_fingerprint": fingerprint(value)}


def validate_flip_cohort(cohort: Mapping[str, Any]) -> None:
    """Refuse changed or unsupported private cohort data before deriving any outcome."""
    if cohort.get("schema_version") != _SCHEMA or cohort.get("cohort_fingerprint") != fingerprint(
        {key: value for key, value in cohort.items() if key != "cohort_fingerprint"}
    ):
        msg = "semantic flip cohort fingerprint or schema does not match the frozen data"
        raise ValueError(msg)


def hypergeometric_count_interval(population: int, sample: int, observed: int) -> tuple[int, int]:
    """Invert exact equal-tailed 95% hypergeometric tests for the population count.

    Args:
        population: Finite trial population size.
        sample: SRS size without replacement.
        observed: Binary successes among sampled trial occurrences.

    Returns:
        Inclusive integer lower and upper counts. Each excluded tail has probability
        below 0.025; discreteness can make actual coverage exceed 95%.

    Raises:
        ValueError: If sample, population or observed count is outside its integer range.
    """
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (population, sample, observed)
        )
        or not 0 <= observed <= sample <= population
        or sample == 0
    ):
        msg = "hypergeometric counts require 0 <= observed <= sample <= population and sample > 0"
        raise ValueError(msg)
    low, high = observed, population - sample + observed
    left, right = low, high
    while left < right:
        middle = (left + right) // 2
        if float(hypergeom.sf(observed - 1, population, middle, sample)) >= _TAIL_PROBABILITY:
            right = middle
        else:
            left = middle + 1
    lower = left
    left, right = low, high
    while left < right:
        middle = (left + right + 1) // 2
        if float(hypergeom.cdf(observed, population, middle, sample)) >= _TAIL_PROBABILITY:
            left = middle
        else:
            right = middle - 1
    return lower, left


def _bounds(
    rows: list[dict[str, Any]], statuses: Mapping[str, str], attempted: int
) -> dict[str, Any]:
    """Count only observed consensus and retain every attempted-trial denominator."""
    valid = sum(statuses.get(row["pair_id"]) == "valid" for row in rows)
    invalid = sum(statuses.get(row["pair_id"]) == "invalid" for row in rows)
    unknown = len(rows) - valid - invalid
    return {
        "attempted_trials": attempted,
        "classifier_flip_trials": len(rows),
        "confirmed_valid_flip_trials": valid,
        "confirmed_invalid_flip_trials": invalid,
        "unresolved_flip_trials": unknown,
        "valid_flip_count_bounds": {"lower": valid, "upper": valid + unknown},
        "valid_flip_yield_bounds": {
            "lower": valid / attempted if attempted else None,
            "upper": (valid + unknown) / attempted if attempted else None,
        },
    }


def _interim(
    cohort: Mapping[str, Any], statuses: Mapping[str, str], *, gated: bool
) -> dict[str, Any]:
    """Use the common changed-trial SRS, treating noncontributing gated trials as zero."""
    trials = cohort["trials"]
    sample_ids = set(cohort["sample_occurrence_ids"])
    sampled = [row for row in trials if row["occurrence_id"] in sample_ids]
    eligible = [row for row in sampled if not gated or row["source_success"]]
    valid = sum(statuses.get(row["pair_id"]) == "valid" for row in eligible)
    unknown = sum(statuses.get(row["pair_id"], "unresolved") == "unresolved" for row in eligible)
    if not sampled:
        lower = upper = 0
    else:
        lower = hypergeometric_count_interval(len(trials), len(sampled), valid)[0]
        upper = hypergeometric_count_interval(len(trials), len(sampled), valid + unknown)[1]
    if gated:
        upper = min(upper, sum(row["source_success"] for row in trials))
    denominator = cohort["attempted_trials"]
    return {
        "sample_trials": len(sampled),
        "sample_confirmed_valid_trials": valid,
        "sample_unknown_trials": unknown,
        "point_estimate": None
        if unknown
        else (len(trials) * valid / len(sampled) / denominator if sampled else 0.0),
        "population_count_interval": {"lower": lower, "upper": upper},
        "yield_interval": {"lower": lower / denominator, "upper": upper / denominator},
        "method": "exact equal-tailed hypergeometric 95% interval; unresolved count envelope",
        "sampling_frame": "all target-changing trial occurrences for both policies",
        "known_gating_count_bound_applied": gated,
    }


def maximum_interval_margin(population: int, sample: int, attempted: int) -> float:
    """Return the largest resolved-sample endpoint distance from the yield estimate."""
    if not population or not sample:
        return 0.0
    return (
        max(
            max(
                abs(bound - population * observed / sample)
                for bound in hypergeometric_count_interval(population, sample, observed)
            )
            for observed in range(sample + 1)
        )
        / attempted
    )


def _association(cells: list[dict[str, Any]], similarity: pd.DataFrame | None) -> dict[str, Any]:
    """Calculate a designed-pool association only when every contributing flip resolves."""
    reason = "Human validity remains unresolved"
    if not any(row["unresolved_flip_trials"] for row in cells) and similarity is not None:
        frame = pd.DataFrame(cells)
        means = frame.groupby("surrogate")["valid_flip_yield_bounds"].apply(
            lambda values: float(np.mean([value["lower"] for value in values]))
        )
        cka = similarity.set_index("surrogate")["mean_cka"]
        if similarity["surrogate"].duplicated().any() or not set(means.index).issubset(cka.index):
            msg = "validity association requires unique CKA values for every retained surrogate"
            raise ValueError(msg)
        if (
            len(means) >= _MIN_ASSOCIATION_UNITS
            and means.nunique() > 1
            and cka.loc[means.index].nunique() > 1
        ):
            return {
                **spearman_association(cka.loc[means.index].tolist(), means.tolist()),
                "unit": "designed surrogate pool; macro recipe yield",
                "null": "exchangeability within the designed pool",
            }
        reason = "Spearman association is undefined for constant or insufficient inputs"
    elif similarity is None:
        reason = "CKA summaries were not supplied"
    return {
        "rho": None,
        "two_sided_p": None,
        "reason": reason,
        "n": len({row["surrogate"] for row in cells}),
    }


def summarize_flip_review(
    cohort: Mapping[str, Any], consensus: pd.DataFrame, similarity: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Report separate interim inference and literal census bounds without imputing labels.

    Args:
        cohort: Frozen private cohort produced from saved pairs and predictions.
        consensus: Observed human consensus; absent pairs remain unresolved.
        similarity: Optional historical CKA summaries for fully resolved associations.

    Returns:
        Aggregate-only coverage, policy intervals, census bounds, cell counts and
        distinct-original union bounds. Interim sample inference does not estimate
        per-cell validity or original-union validity.

    Raises:
        ValueError: If cohort integrity or consensus membership/statuses fail validation.
    """
    validate_flip_cohort(cohort)
    dataset = cohort["review_dataset"]
    ids = {pair["pair_id"] for pair in dataset["pairs"]} if dataset else set()
    if (
        not {"pair_id", "status"}.issubset(consensus)
        or consensus["pair_id"].duplicated().any()
        or not set(consensus["pair_id"]).issubset(ids)
        or not consensus["status"].isin(["valid", "invalid", "unresolved"]).all()
    ):
        msg = (
            "consensus requires unique cohort pairs and explicit valid/invalid/unresolved statuses"
        )
        raise ValueError(msg)
    statuses = dict(zip(consensus["pair_id"], consensus["status"], strict=True))
    resolved = sum(value in {"valid", "invalid"} for value in statuses.values())
    reviewed = int((consensus.get("reviewer_count", pd.Series(dtype=int)) > 0).sum())
    complete = int(
        (
            consensus.get("complete_review_count", pd.Series(dtype=int)) == _MIN_ASSOCIATION_UNITS
        ).sum()
    )
    result: dict[str, Any] = {
        "schema_version": "semantic-flip-review-v1",
        "review_preparation_id": fingerprint(
            ["public-semantic-flip-preparation-v1", cohort["cohort_fingerprint"]]
        ),
        "sampling_design": (
            "SRS without replacement of changed trial occurrences; "
            "pair deduplication after sampling"
        ),
        "review_status": "complete"
        if complete == len(ids)
        else ("in_progress" if reviewed or resolved else "pending"),
        "validity_status": "census_resolved"
        if resolved == len(ids)
        else ("partially_resolved" if resolved else "unresolved"),
        "attempted_trials": cohort["attempted_trials"],
        "classifier_flip_trials": len(cohort["trials"]),
        "distinct_pair_workload": len(ids),
        "resolved_pairs": resolved,
        "reviewed_pairs": reviewed,
        "two_reviewer_complete_pairs": complete,
        "unresolved_pairs": len(ids) - resolved,
        "stage_one_sample_trials": len(cohort["sample_occurrence_ids"]),
        "stage_one_distinct_pairs": dataset["stage_one_pair_count"] if dataset else 0,
        "resolved_sample_maximum_interval_margin": maximum_interval_margin(
            len(cohort["trials"]), len(cohort["sample_occurrence_ids"]), cohort["attempted_trials"]
        ),
        "margin_is_design_property": True,
        "interim_cell_rates_estimated": False,
        "interim_original_union_estimated": False,
        "census_bounds_are_confidence_intervals": False,
        "policies": {},
        "cells": [],
        "surrogates": [],
        "recipes": [],
    }
    for name, gated in [("all_returned_candidates", False), ("source_success_gated", True)]:
        rows = [row for row in cohort["trials"] if not gated or row["source_success"]]
        policy = _bounds(rows, statuses, cohort["attempted_trials"])
        policy["interim"] = _interim(cohort, statuses, gated=gated)
        valid_originals = {
            row["original_id"] for row in rows if statuses.get(row["pair_id"]) == "valid"
        }
        possible_originals = {
            row["original_id"]
            for row in rows
            if statuses.get(row["pair_id"], "unresolved") != "invalid"
        }
        originals = len(cohort["original_ids"])
        policy["original_union"] = {
            "originals": originals,
            "target_original_positive_originals": cohort["target_original_positive_originals"],
            "valid_original_count_bounds": {
                "lower": len(valid_originals),
                "upper": len(possible_originals),
            },
            "coverage_bounds": {
                "lower": len(valid_originals) / originals,
                "upper": len(possible_originals) / originals,
            },
            "point_estimate": len(valid_originals) / originals
            if valid_originals == possible_originals
            else None,
        }
        cells = [
            {
                "policy": name,
                "surrogate": cell["surrogate"],
                "recipe": cell["recipe"],
                **_bounds(
                    [
                        row
                        for row in rows
                        if (row["surrogate"], row["recipe"]) == (cell["surrogate"], cell["recipe"])
                    ],
                    statuses,
                    cell["attempted"],
                ),
            }
            for cell in cohort["cells"]
        ]
        policy["association"] = _association(cells, similarity)
        result["policies"][name] = policy
        result["cells"].extend(cells)
        for field, output in [("surrogate", "surrogates"), ("recipe", "recipes")]:
            for identity in sorted({cell[field] for cell in cohort["cells"]}):
                denominator = sum(
                    cell["attempted"] for cell in cohort["cells"] if cell[field] == identity
                )
                result[output].append(
                    {
                        "policy": name,
                        field: identity,
                        **_bounds(
                            [row for row in rows if row[field] == identity], statuses, denominator
                        ),
                    }
                )
    return result
