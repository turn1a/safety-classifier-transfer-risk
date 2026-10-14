"""Prepare private blinded review and aggregate coverage from saved text pairs.

Edit distances describe text changes. They do not establish preserved meaning or
injection intent. Only two independent human reviewers can resolve those labels.
This module performs no I/O, inference, model loading, or network access.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from transfer_risk.lib.seeds import derive_seeds

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_STRATUM = ("source", "surrogate", "recipe")
_LABELS = ("semantic_preserved", "intent_preserved", "original_is_injection")
_COUNT_COLUMNS = (
    "attempt_count",
    "source_successful_count",
    "source_failed_count",
    "source_skipped_count",
    "source_other_count",
)
_FIDELITY_NUMERIC = (
    "original_characters",
    "perturbed_characters",
    "character_edit_distance",
    "character_distance_fraction",
    "original_tokens",
    "perturbed_tokens",
    "token_edit_distance",
    "token_distance_fraction",
)
_FIDELITY_FLAGS = ("identical", "whitespace_only_change", "empty_perturbation")
_PAIR_COLUMNS = (
    "pair_id",
    "original_id",
    "original",
    "perturbed",
    *_FIDELITY_NUMERIC,
    *_FIDELITY_FLAGS,
)
_LINK_COLUMNS = ("pair_id", "original_id", *_STRATUM, *_COUNT_COLUMNS)
_SAMPLING_COLUMNS = (
    "stratum_population_pairs",
    "stratum_drawn_pairs",
    "stratum_draw_probability",
    "drawn_in_stratum",
    "selected_for_review",
    "inclusion_probability",
    "design_weight",
)
_ANNOTATION_COLUMNS = (
    "pair_id",
    "original",
    "perturbed",
    "reviewer_id",
    "independent_review",
    *_LABELS,
    "notes",
    "review_status",
)
_CONSENSUS_COLUMNS = (
    "pair_id",
    "status",
    "reason",
    "reviewer_count",
    "complete_review_count",
    *_LABELS,
)
_TWO_REVIEWERS = 2
_CELL_KEY_PARTS = 2


def _stable_id(values: Sequence[str], namespace: str) -> str:
    """Hash length-unambiguous exact Unicode strings within a named namespace."""
    payload = json.dumps([namespace, *values], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_pair_id(original: str, perturbed: str) -> str:
    """Return a stable SHA-256 identifier for an exact, ordered text pair.

    Args:
        original: Saved original text, without normalization or further clipping.
        perturbed: Saved perturbed text, without normalization.

    Returns:
        Private hexadecimal pair identifier. Hashing does not make prompt text public-safe.
    """
    return _stable_id((original, perturbed), "semantic-audit-pair-v1")


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    """Compute exact unit-cost Levenshtein distance with a bit-vector recurrence."""
    prefix = 0
    while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]:
        prefix += 1
    left, right = left[prefix:], right[prefix:]
    suffix = 0
    while suffix < min(len(left), len(right)) and left[-suffix - 1] == right[-suffix - 1]:
        suffix += 1
    if suffix:
        left, right = left[:-suffix], right[:-suffix]
    if len(left) > len(right):
        left, right = right, left
    if not left:
        return len(right)
    masks: dict[str, int] = defaultdict(int)
    for index, symbol in enumerate(left):
        masks[symbol] |= 1 << index
    positive = (1 << len(left)) - 1
    negative = 0
    distance = len(left)
    high_bit = 1 << (len(left) - 1)
    for symbol in right:
        equal = masks.get(symbol, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal | negative
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        distance += bool(positive_horizontal & high_bit) - bool(negative_horizontal & high_bit)
        positive_horizontal = (positive_horizontal << 1) | 1
        negative_horizontal <<= 1
        positive = negative_horizontal | ~(vertical | positive_horizontal)
        negative = horizontal & positive_horizontal
    return distance


def pair_fidelity(original: str, perturbed: str) -> dict[str, int | float | bool]:
    """Measure exact text edits without classifying their semantic validity.

    Args:
        original: Exact saved original text.
        perturbed: Exact saved perturbation.

    Returns:
        Unicode-codepoint and whitespace-token Levenshtein distances, lengths,
        distances divided by the larger length, and exact text-change flags.
        A distance fraction is zero for two empty sequences.
    """
    original_tokens, perturbed_tokens = original.split(), perturbed.split()
    character_distance = _edit_distance(original, perturbed)
    token_distance = _edit_distance(original_tokens, perturbed_tokens)
    return {
        "original_characters": len(original),
        "perturbed_characters": len(perturbed),
        "character_edit_distance": character_distance,
        "character_distance_fraction": character_distance / max(len(original), len(perturbed), 1),
        "original_tokens": len(original_tokens),
        "perturbed_tokens": len(perturbed_tokens),
        "token_edit_distance": token_distance,
        "token_distance_fraction": token_distance
        / max(len(original_tokens), len(perturbed_tokens), 1),
        "identical": original == perturbed,
        "whitespace_only_change": original != perturbed
        and "".join(original_tokens) == "".join(perturbed_tokens),
        "empty_perturbation": not perturbed,
    }


def _validated_record(
    record: Mapping[str, Any], source_map: Mapping[str, str]
) -> tuple[str, str, str]:
    """Validate saved text and status fields without printing private text on errors."""
    original, perturbed = record.get("original"), record.get("perturbed")
    if not isinstance(original, str) or not isinstance(perturbed, str):
        msg = "saved attack original and perturbed fields must be strings"
        raise ValueError(msg)
    if original not in source_map:
        msg = "saved attack original is missing from the historical evaluation source mapping"
        raise ValueError(msg)
    if not isinstance(record.get("success"), bool):
        msg = "saved attack success must be an explicit boolean"
        raise ValueError(msg)
    status = record.get("result_type")
    if not isinstance(status, str) or not status:
        msg = "saved attack result_type must be a nonempty string"
        raise ValueError(msg)
    if record["success"] != (status == "SuccessfulAttackResult"):
        msg = "saved attack success contradicts its result_type"
        raise ValueError(msg)
    return original, perturbed, status


def build_pair_population(
    adversarial_examples: Mapping[str, Sequence[Mapping[str, Any]]],
    source_map: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate all saved attempts into exact pairs and private occurrence linkage.

    Args:
        adversarial_examples: All saved cells, including successful, failed, and skipped attempts.
        source_map: Exact historical attack-time originals mapped to their saved corpus source.

    Returns:
        Private text-pair population and private source/surrogate/recipe linkage.
        Repeated attempts contribute multiplicity to linkage, rather than extra review units.

    Raises:
        ValueError: If a cell, record, or historical source mapping is invalid.
    """
    pairs: dict[str, dict[str, Any]] = {}
    linkage: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for cell in sorted(adversarial_examples):
        identity = cell.split("__")
        if len(identity) != _CELL_KEY_PARTS or not all(identity):
            msg = "attack cell keys must have surrogate__recipe form"
            raise ValueError(msg)
        surrogate, recipe = identity
        for record in adversarial_examples[cell]:
            original, perturbed, status = _validated_record(record, source_map)
            pair_id = stable_pair_id(original, perturbed)
            original_id = _stable_id((original,), "semantic-audit-original-v1")
            if pair_id not in pairs:
                pairs[pair_id] = {
                    "pair_id": pair_id,
                    "original_id": original_id,
                    "original": original,
                    "perturbed": perturbed,
                    **pair_fidelity(original, perturbed),
                }
            source = source_map[original]
            key = (pair_id, source, surrogate, recipe)
            if key not in linkage:
                linkage[key] = {
                    "pair_id": pair_id,
                    "original_id": original_id,
                    "source": source,
                    "surrogate": surrogate,
                    "recipe": recipe,
                    **dict.fromkeys(_COUNT_COLUMNS, 0),
                }
            row = linkage[key]
            row["attempt_count"] += 1
            status_column = {
                "SuccessfulAttackResult": "source_successful_count",
                "FailedAttackResult": "source_failed_count",
                "SkippedAttackResult": "source_skipped_count",
            }.get(status, "source_other_count")
            row[status_column] += 1
    return (
        pd.DataFrame([pairs[key] for key in sorted(pairs)], columns=list(_PAIR_COLUMNS)),
        pd.DataFrame([linkage[key] for key in sorted(linkage)], columns=list(_LINK_COLUMNS)),
    )


def sample_annotation_candidates(
    pairs: pd.DataFrame,
    linkage: pd.DataFrame,
    *,
    pairs_per_stratum: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample blinded exact pairs independently within source/surrogate/recipe strata.

    Args:
        pairs: Private globally deduplicated pair population.
        linkage: Private occurrence linkage, with each pair listed once per stratum.
        pairs_per_stratum: Positive fixed draw count, capped at each stratum's size.
        seed: Root seed; the NumPy stream is derived through the project seed helper.

    Returns:
        Private annotation template and private full-population sampling linkage.
        A pair selected in several strata appears once in the template. Its inclusion
        probability is one minus the product of stratum nonselection probabilities.
        Success and target outcomes are never consulted by the sampler.

    Raises:
        ValueError: If draw size is not a positive integer or linkage is inconsistent.
    """
    if (
        isinstance(pairs_per_stratum, bool)
        or not isinstance(pairs_per_stratum, int)
        or pairs_per_stratum < 1
    ):
        msg = "pairs_per_stratum must be a positive integer"
        raise ValueError(msg)
    if linkage.duplicated(["pair_id", *_STRATUM]).any() or set(linkage["pair_id"]) != set(
        pairs["pair_id"]
    ):
        msg = "sampling linkage must cover each pair exactly once per stratum"
        raise ValueError(msg)
    sampled = linkage.sort_values([*_STRATUM, "pair_id"]).reset_index(drop=True).copy()
    for column in _SAMPLING_COLUMNS:
        sampled[column] = False if column in {"drawn_in_stratum", "selected_for_review"} else 0.0
    generator = np.random.default_rng(derive_seeds(seed).numpy)
    nonselection: dict[str, float] = dict.fromkeys(pairs["pair_id"], 1.0)
    selected: set[str] = set()
    for _, group in sampled.groupby(list(_STRATUM), sort=True):
        population_size = len(group)
        draw_size = min(pairs_per_stratum, population_size)
        probability = draw_size / population_size
        drawn = set(generator.choice(group["pair_id"].tolist(), size=draw_size, replace=False))
        selected.update(drawn)
        sampled.loc[group.index, "stratum_population_pairs"] = population_size
        sampled.loc[group.index, "stratum_drawn_pairs"] = draw_size
        sampled.loc[group.index, "stratum_draw_probability"] = probability
        sampled.loc[group.index, "drawn_in_stratum"] = group["pair_id"].isin(drawn)
        for pair_id in group["pair_id"]:
            nonselection[pair_id] *= 1 - probability
    sampled["selected_for_review"] = sampled["pair_id"].isin(selected)
    sampled["inclusion_probability"] = sampled["pair_id"].map(
        {key: 1 - value for key, value in nonselection.items()}
    )
    sampled["design_weight"] = 1 / sampled["inclusion_probability"]
    candidates = pairs.loc[
        pairs["pair_id"].isin(selected), ["pair_id", "original", "perturbed"]
    ].copy()
    for column in _ANNOTATION_COLUMNS[3:]:
        candidates[column] = "pending" if column == "review_status" else ""
    return candidates.sort_values("pair_id").reset_index(drop=True), sampled


def _numeric_summary(values: pd.Series[Any]) -> dict[str, int | float | None]:
    """Return JSON-safe descriptive summaries with an explicit zero-observation case."""
    if values.empty:
        return {"count": 0, "mean": None, "median": None, "p95": None, "maximum": None}
    numeric = values.astype(float)
    return {
        "count": len(numeric),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "p95": float(numeric.quantile(0.95)),
        "maximum": float(numeric.max()),
    }


def _stratum_counts(linkage: pd.DataFrame) -> list[dict[str, Any]]:
    """Aggregate non-text occurrence and sampling coverage in the designed strata."""
    rows: list[dict[str, Any]] = []
    for keys, group in linkage.groupby(list(_STRATUM), sort=True):
        row = dict(zip(_STRATUM, keys, strict=True))
        row.update({column: int(group[column].sum()) for column in _COUNT_COLUMNS})
        row["unique_pairs"] = len(group)
        if "selected_for_review" in group:
            row["sampled_pairs"] = int(group["selected_for_review"].sum())
            row["direct_stratum_draws"] = int(group["drawn_in_stratum"].sum())
        rows.append(row)
    return rows


def fidelity_summary(pairs: pd.DataFrame, linkage: pd.DataFrame) -> dict[str, Any]:
    """Create aggregate-only mechanical fidelity descriptions for the full saved pool.

    Args:
        pairs: Private exact-pair population with mechanical edit measurements.
        linkage: Private occurrence and optional sampling linkage.

    Returns:
        Public counts, distance summaries, and source/surrogate/recipe aggregates.
        Neither raw text, record identifiers, nor individual labels are included.
    """
    attempts = int(linkage["attempt_count"].sum())
    return {
        "schema_version": "semantic-audit-v1",
        "population": {
            "attempts": attempts,
            "unique_pairs": len(pairs),
            "unique_originals": int(pairs["original_id"].nunique()),
            "duplicate_attempts": attempts - len(pairs),
            **{flag: int(pairs[flag].sum()) for flag in _FIDELITY_FLAGS},
        },
        "distance_unit": "globally deduplicated exact original/perturbation pair",
        "character_unit": "Unicode codepoint, without normalization",
        "tokenization": "Python str.split on whitespace; punctuation remains attached",
        "distance_definition": (
            "unit-cost Levenshtein insertions, deletions and substitutions; "
            "divided by the larger sequence length for fractions"
        ),
        "distance_summaries": {
            column: _numeric_summary(pairs[column]) for column in _FIDELITY_NUMERIC
        },
        "strata": _stratum_counts(linkage),
        "semantic_validity_established": False,
        "interpretation": (
            "Edit counts and distance fractions describe saved text changes. "
            "They do not establish preserved meaning, preserved injection intent, "
            "or a correct original injection label."
        ),
    }


def _clean_field(value: Any) -> str:
    """Normalize annotation fields, treating CSV missing values as pending."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if not isinstance(value, str):
        msg = "annotation labels and reviewer identities must be explicit strings"
        raise ValueError(msg)
    return value.strip().casefold()


def _reviewer_identities(reviewer_ids: Sequence[str], *, submitted: bool) -> set[str]:
    """Require two distinct registered identities whenever review data are submitted."""
    if isinstance(reviewer_ids, str):
        msg = "semantic review requires two distinct nonempty registered reviewer identities"
        raise ValueError(msg)
    identities = {_clean_field(value) for value in reviewer_ids}
    if (reviewer_ids or submitted) and (
        len(reviewer_ids) != _TWO_REVIEWERS or len(identities) != _TWO_REVIEWERS or "" in identities
    ):
        msg = "semantic review requires two distinct nonempty registered reviewer identities"
        raise ValueError(msg)
    return identities


def _validated_annotations(
    annotations: pd.DataFrame,
    pair_ids: set[str],
    reviewer_ids: Sequence[str],
) -> dict[str, list[dict[str, str]]]:
    """Validate identities, label vocabulary, sample membership, and independence."""
    records = cast("list[dict[str, Any]]", annotations.to_dict("records"))
    submitted = any(
        _clean_field(row.get("reviewer_id"))
        or any(_clean_field(row.get(label)) for label in _LABELS)
        for row in records
    )
    identities = _reviewer_identities(reviewer_ids, submitted=submitted)
    responses: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in records:
        pair_id = str(row.get("pair_id", ""))
        if pair_id not in pair_ids:
            msg = "annotation pair_id must belong to the selected review sample"
            raise ValueError(msg)
        reviewer = _clean_field(row.get("reviewer_id"))
        labels = {label: _clean_field(row.get(label)) for label in _LABELS}
        if not reviewer and not any(labels.values()):
            continue
        if reviewer not in identities:
            msg = "each submitted annotation must name a registered reviewer"
            raise ValueError(msg)
        if (pair_id, reviewer) in seen:
            msg = "duplicate review from one reviewer for the same pair"
            raise ValueError(msg)
        seen.add((pair_id, reviewer))
        if any(value not in {"", "yes", "no", "uncertain"} for value in labels.values()):
            msg = "review labels must be yes, no, uncertain, or empty while pending"
            raise ValueError(msg)
        independent = _clean_field(row.get("independent_review"))
        if independent not in {"", "yes"}:
            msg = "submitted independent_review must explicitly attest yes or remain pending"
            raise ValueError(msg)
        responses[pair_id].append({**labels, "independent_review": independent})
    return responses


def _resolve_pair(pair_id: str, responses: list[dict[str, str]]) -> dict[str, Any]:
    """Resolve the validity conjunction separately from review-field completeness."""
    independent = [row for row in responses if row["independent_review"] == "yes"]
    complete = [row for row in independent if all(row[label] for label in _LABELS)]
    row: dict[str, Any] = {
        "pair_id": pair_id,
        "status": "unresolved",
        "reason": "pending",
        "reviewer_count": len(responses),
        "complete_review_count": len(complete),
        **dict.fromkeys(_LABELS, ""),
    }
    if len(independent) == _TWO_REVIEWERS:
        row.update(
            {
                label: independent[0][label]
                if independent[0][label] == independent[1][label]
                else ""
                for label in _LABELS
            }
        )
    if len(independent) == _TWO_REVIEWERS and any(
        all(response[label] == "no" for response in independent) for label in _LABELS
    ):
        row["status"] = "invalid"
        row["reason"] = "agreed_no_criterion"
    elif len(complete) != _TWO_REVIEWERS:
        row["reason"] = "incomplete" if responses else "pending"
    elif any(response[label] == "uncertain" for response in complete for label in _LABELS):
        row["reason"] = "uncertain"
    elif any(complete[0][label] != complete[1][label] for label in _LABELS):
        row["reason"] = "disagreement"
    else:
        row.update({label: complete[0][label] for label in _LABELS})
        row["status"] = "valid"
        row["reason"] = "two_reviewer_agreement"
    return row


def resolve_annotations(
    annotations: pd.DataFrame,
    selected_pair_ids: Sequence[str],
    reviewer_ids: Sequence[str],
) -> pd.DataFrame:
    """Resolve private human labels conservatively for exactly the selected pairs.

    Args:
        annotations: Private reviewer rows. Empty or untouched templates remain pending.
        selected_pair_ids: Globally deduplicated sampled pair identifiers.
        reviewer_ids: Exactly two distinct registered identities when labels are submitted.

    Returns:
        Private pair consensus. A pair is valid only when both independent reviewers
        answer yes to all three criteria. An agreed no on any one criterion makes the
        conjunction invalid, even if other criteria are incomplete or disputed. All
        remaining cases are unresolved. Complete-review counts still require every field.

    Raises:
        ValueError: If identities, duplicate rows, sample membership, labels,
            or attestations fail validation.
    """
    if len(set(selected_pair_ids)) != len(selected_pair_ids):
        msg = "selected_pair_ids must be deduplicated"
        raise ValueError(msg)
    responses = _validated_annotations(annotations, set(selected_pair_ids), reviewer_ids)
    return pd.DataFrame(
        [
            _resolve_pair(pair_id, responses.get(pair_id, []))
            for pair_id in sorted(selected_pair_ids)
        ],
        columns=list(_CONSENSUS_COLUMNS),
    )


def _bounds(valid: int, invalid: int, population: int) -> dict[str, float | None]:
    """Bound full finite-population validity using only actually resolved labels."""
    return {
        "lower": valid / population if population else None,
        "upper": (population - invalid) / population if population else None,
    }


def _weighted_sample_status(
    linkage: pd.DataFrame, statuses: Mapping[str, str]
) -> dict[str, float | None]:
    """Describe inverse-inclusion-weighted sample mass without imputing missing labels."""
    sampled = linkage.loc[linkage["selected_for_review"]].drop_duplicates("pair_id")
    denominator = float(sampled["design_weight"].sum())
    shares: dict[str, float | None] = {}
    for status in ("valid", "invalid", "unresolved"):
        numerator = float(
            sampled.loc[sampled["pair_id"].map(statuses) == status, "design_weight"].sum()
        )
        shares[status] = numerator / denominator if denominator else None
    return shares


def review_summary(
    pairs: pd.DataFrame,
    linkage: pd.DataFrame,
    consensus: pd.DataFrame,
) -> dict[str, Any]:
    """Publish review coverage and finite-population bounds without label imputation.

    Args:
        pairs: Private full exact-pair population.
        linkage: Private full occurrence and sampling linkage.
        consensus: Validated private consensus for every selected pair.

    Returns:
        Aggregate-only coverage, literal finite-population validity bounds, and
        inverse-inclusion-weighted sample shares. A Hajek pair-validity point estimate
        is available only when every selected pair is resolved. No confidence interval,
        cell-rate estimate, or CKA association is inferred from this review design.

    Raises:
        ValueError: If consensus does not cover exactly the selected sample.
    """
    selected = set(linkage.loc[linkage["selected_for_review"], "pair_id"])
    if consensus["pair_id"].duplicated().any() or set(consensus["pair_id"]) != selected:
        msg = "consensus must cover exactly the selected review sample"
        raise ValueError(msg)
    if not set(consensus["status"]).issubset({"valid", "invalid", "unresolved"}):
        msg = "consensus status must be valid, invalid, or unresolved"
        raise ValueError(msg)
    statuses = dict(zip(consensus["pair_id"], consensus["status"], strict=True))
    valid = int((consensus["status"] == "valid").sum())
    invalid = int((consensus["status"] == "invalid").sum())
    attempt_status = linkage["pair_id"].map(statuses).fillna("unresolved")
    attempt_valid = int(linkage.loc[attempt_status == "valid", "attempt_count"].sum())
    attempt_invalid = int(linkage.loc[attempt_status == "invalid", "attempt_count"].sum())
    attempts = int(linkage["attempt_count"].sum())
    weighted = _weighted_sample_status(linkage, statuses)
    resolved = valid + invalid
    reviewed = int((consensus["reviewer_count"] > 0).sum())
    status = "pending" if not reviewed else "review_in_progress"
    if resolved:
        status = "sample_resolved" if resolved == len(selected) else "partial"
    return {
        "schema_version": "semantic-audit-v1",
        "status": status,
        "population_pairs": len(pairs),
        "population_attempts": attempts,
        "sampled_pairs": len(selected),
        "reviewed_pairs": reviewed,
        "two_reviewer_complete_pairs": int(
            (consensus["complete_review_count"] == _TWO_REVIEWERS).sum()
        ),
        "resolved_pairs": resolved,
        "confirmed_valid_pairs": valid,
        "confirmed_invalid_pairs": invalid,
        "unresolved_sample_pairs": len(selected) - resolved,
        "unresolved_population_pairs": len(pairs) - resolved,
        "population_resolution_coverage": resolved / len(pairs) if len(pairs) else None,
        "sample_resolution_coverage": resolved / len(selected) if selected else None,
        "finite_population_pair_validity_bounds": _bounds(valid, invalid, len(pairs)),
        "finite_population_attempt_validity_bounds": _bounds(
            attempt_valid, attempt_invalid, attempts
        ),
        "bounds_are_confidence_intervals": False,
        "weighted_sample_status_shares": weighted,
        "design_weighted_validity_estimate": weighted["valid"]
        if selected and resolved == len(selected)
        else None,
        "point_estimate_method": (
            "Hajek ratio of inverse union-inclusion probabilities; unique exact pairs"
        ),
        "sampling_design": (
            "independent simple random sampling without replacement within "
            "source/surrogate/recipe strata, followed by pair deduplication; "
            "pi = 1 - product(1 - n_h/N_h)"
        ),
        "cell_validity_rates_estimated": False,
        "cka_validity_association": None,
        "strata": _stratum_counts(linkage),
        "limitations": [
            (
                "A pair is valid after two independent yes answers to every criterion and "
                "invalid after an agreed no on any one criterion. Other missing answers, "
                "uncertainty and disagreement remain unresolved; review completeness is separate."
            ),
            (
                "Finite-population bounds use observed labels only. Weighted sample shares "
                "and a fully resolved sample point estimate carry sampling uncertainty; "
                "no design-based confidence interval is supplied."
            ),
            (
                "The sample supports overall pair review preparation. It does not establish "
                "precise validity rates for 50 surrogate/recipe cells or a "
                "CKA-validity association."
            ),
            (
                "Distinct registered reviewer identities and independence attestations are "
                "checked mechanically; a human coordinator must enforce separate review "
                "before discussion."
            ),
            (
                "Meaning, injection intent and the original injection label require human "
                "judgment. Edit distances provide no semantic-validity label."
            ),
        ],
    }
