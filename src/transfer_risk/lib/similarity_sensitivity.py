"""Pure summaries for the training-only CKA sensitivity analysis.

These helpers compare saved original CKA evidence with a CKA probe sampled solely from the
existing local training split. They never load models, inspect attack records, or perform I/O.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from transfer_risk.lib.ablation import selection_ablation
from transfer_risk.lib.association import spearman_association
from transfer_risk.lib.dbs import diagonal_box_similarity
from transfer_risk.lib.seeds import derive_seeds

if TYPE_CHECKING:
    import pandas as pd

_SIMILARITY_COLUMNS = ("surrogate", "mean_cka", "dbs")
_SELECTION_GROUPS = ("M1", "M2")
_TRUE_FLIP_ABLATION_COMPONENT = 2


def export_training_probe_similarity(similarity: pd.DataFrame) -> pd.DataFrame:
    """Select the safe public columns from training-probe similarity results.

    Args:
        similarity: One row per surrogate with mean CKA and corrected DBS.

    Returns:
        A copy containing only ``surrogate``, ``mean_cka``, and ``dbs``.

    Raises:
        ValueError: If the similarity table lacks a required public column.
    """
    _require_columns(similarity, _SIMILARITY_COLUMNS, artifact="training-probe similarity")
    return similarity.loc[:, list(_SIMILARITY_COLUMNS)].copy()


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, artifact: str) -> None:
    """Raise when a dataframe is missing required columns.

    Args:
        frame: Dataframe to validate.
        columns: Required column names.
        artifact: Human-readable artifact name for error messages.

    Raises:
        ValueError: If any required column is absent.
    """
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        msg = f"{artifact} is missing required columns: {missing}"
        raise ValueError(msg)


def _finite_float(value: Any, *, field: str) -> float:
    """Convert one scalar to a finite float.

    Args:
        value: Candidate numeric value.
        field: Human-readable field name for error messages.

    Returns:
        Finite floating-point value.

    Raises:
        ValueError: If the value is boolean, non-numeric, or non-finite.
    """
    if isinstance(value, bool):
        msg = f"{field} must be a finite number"
        raise ValueError(msg)
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        msg = f"{field} must be a finite number"
        raise ValueError(msg) from error
    if not math.isfinite(number):
        msg = f"{field} must be a finite number"
        raise ValueError(msg)
    return number


def _optional_finite_float(value: Any, *, field: str) -> float | None:
    """Convert a nullable scalar to a finite float.

    Args:
        value: Candidate nullable numeric value.
        field: Human-readable field name for error messages.

    Returns:
        ``None`` for null values, otherwise a finite floating-point value.
    """
    if value is None:
        return None
    return _finite_float(value, field=field)


def _metric_by_surrogate(
    similarity: pd.DataFrame,
    *,
    metric: str,
    artifact: str,
) -> dict[str, float]:
    """Extract one finite scalar similarity metric per unique surrogate.

    Args:
        similarity: Similarity dataframe.
        metric: Metric column to extract.
        artifact: Human-readable artifact name for error messages.

    Returns:
        Mapping from surrogate name to finite metric value.

    Raises:
        ValueError: If the dataframe lacks columns, has blank/duplicate names, or has an
            undefined metric value.
    """
    _require_columns(similarity, ("surrogate", metric), artifact=artifact)
    values: dict[str, float] = {}
    for raw_name, raw_value in zip(
        similarity["surrogate"].tolist(),
        similarity[metric].tolist(),
        strict=True,
    ):
        name = str(raw_name)
        if not name:
            msg = f"{artifact} contains a blank surrogate name"
            raise ValueError(msg)
        if name in values:
            msg = f"{artifact} contains duplicate surrogate {name!r}"
            raise ValueError(msg)
        values[name] = _finite_float(raw_value, field=f"{artifact} {metric} for {name!r}")
    if not values:
        msg = f"{artifact} must contain at least one surrogate"
        raise ValueError(msg)
    return values


def _aligned_values(
    first: Mapping[str, float],
    second: Mapping[str, float],
    *,
    first_name: str,
    second_name: str,
) -> tuple[list[str], list[float], list[float]]:
    """Align two surrogate metrics and reject mismatched configured pools.

    Args:
        first: First metric by surrogate.
        second: Second metric by surrogate.
        first_name: Label for the first metric in errors.
        second_name: Label for the second metric in errors.

    Returns:
        Sorted surrogate names and aligned first/second metric vectors.

    Raises:
        ValueError: If the two mappings do not cover exactly the same surrogate names.
    """
    first_names = set(first)
    second_names = set(second)
    if first_names != second_names:
        only_first = sorted(first_names.difference(second_names))
        only_second = sorted(second_names.difference(first_names))
        msg = (
            f"{first_name} and {second_name} must use the same surrogate pool; "
            f"only {first_name}={only_first}, only {second_name}={only_second}"
        )
        raise ValueError(msg)
    names = sorted(first)
    return names, [first[name] for name in names], [second[name] for name in names]


def _association_block(
    first: Sequence[float],
    second: Sequence[float],
    *,
    pool_label: str,
) -> dict[str, Any]:
    """Return a labelled Spearman comparison with an exchangeability-null p-value.

    Args:
        first: First aligned metric values.
        second: Second aligned metric values.
        pool_label: Description of the designed surrogate pool.

    Returns:
        JSON-compatible association metadata and either an estimated or not-estimable
        exchangeability-null result.
    """
    designed_pool = {
        "label": pool_label,
        "unit": "one row per surrogate",
        "n_surrogates": len(first),
    }
    try:
        stats = spearman_association(first, second)
    except ValueError as error:
        return {
            "designed_pool": designed_pool,
            "exchangeability_null": {
                "status": "not_estimable",
                "reason": str(error),
                "rho": None,
                "two_sided_p": None,
                "n": len(first),
                "exact": False,
                "p_value_label": "not estimable for the designed surrogate pool",
            },
        }
    exact = bool(stats["exact"])
    p_value_label = (
        "exact enumeration under the exchangeability null for the designed surrogate pool"
        if exact
        else "normal approximation under the exchangeability null for the designed surrogate pool"
    )
    return {
        "designed_pool": designed_pool,
        "exchangeability_null": {
            "status": "estimated",
            "rho": float(stats["rho"]),
            "two_sided_p": float(stats["two_sided_p"]),
            "n": int(stats["n"]),
            "exact": exact,
            "p_value_label": p_value_label,
        },
    }


def _selection_members(
    selection: Mapping[str, Any],
    *,
    artifact: str,
) -> dict[str, list[str]]:
    """Validate and normalize M1/M2 membership from a saved selection artifact.

    Args:
        selection: Mapping containing M1 and M2 surrogate sequences.
        artifact: Human-readable artifact name for error messages.

    Returns:
        Mapping with normalized M1 and M2 lists.

    Raises:
        ValueError: If a group is missing, empty, duplicated, overlapping, or not a sequence.
    """
    normalized: dict[str, list[str]] = {}
    for group in _SELECTION_GROUPS:
        if group not in selection:
            msg = f"{artifact} is missing {group!r}"
            raise ValueError(msg)
        raw_members = selection[group]
        if isinstance(raw_members, (str, bytes, bytearray)) or not isinstance(
            raw_members, Sequence
        ):
            msg = f"{artifact} {group!r} must be a sequence of surrogate names"
            raise ValueError(msg)
        members = [str(member) for member in raw_members]
        if not members or len(members) != len(set(members)):
            msg = f"{artifact} {group!r} must contain unique surrogate names"
            raise ValueError(msg)
        normalized[group] = members
    if set(normalized["M1"]).intersection(normalized["M2"]):
        msg = f"{artifact} M1 and M2 must not overlap"
        raise ValueError(msg)
    return normalized


def _validate_selection_pool(
    selection: Mapping[str, Sequence[str]],
    *,
    surrogate_names: set[str],
    artifact: str,
) -> None:
    """Raise if saved M1/M2 membership names fall outside a similarity table.

    Args:
        selection: Normalized M1/M2 membership.
        surrogate_names: Names present in the corresponding similarity table.
        artifact: Human-readable artifact name for errors.

    Raises:
        ValueError: If either selection group contains an unknown surrogate.
    """
    selected = set(selection["M1"]).union(selection["M2"])
    unknown = sorted(selected.difference(surrogate_names))
    if unknown:
        msg = f"{artifact} contains surrogates absent from its similarity table: {unknown}"
        raise ValueError(msg)


def _membership_overlap(
    original: Mapping[str, Sequence[str]],
    training: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, Any]]:
    """Compute group-specific membership intersections, unions, and Jaccard scores.

    Args:
        original: Original M1/M2 membership.
        training: Training-probe M1/M2 membership.

    Returns:
        Per-group membership overlap metadata.
    """
    overlap: dict[str, dict[str, Any]] = {}
    for group in _SELECTION_GROUPS:
        original_members = list(original[group])
        training_members = list(training[group])
        training_set = set(training_members)
        intersection = [name for name in original_members if name in training_set]
        union = list(dict.fromkeys([*original_members, *training_members]))
        overlap[group] = {
            "intersection": intersection,
            "union": union,
            "intersection_count": len(intersection),
            "union_count": len(union),
            "jaccard": len(intersection) / len(union),
        }
    return overlap


def _probe_metadata(probe: pd.DataFrame) -> dict[str, Any]:
    """Build non-text metadata describing a sampled training-only probe.

    Args:
        probe: Training-only probe dataframe.

    Returns:
        JSON-compatible probe size, canonical-label counts, and source-count metadata.

    Raises:
        ValueError: If label or source metadata are absent.
    """
    _require_columns(probe, ("label", "source"), artifact="training probe")
    label_counts = {
        str(label): int(count)
        for label, count in probe["label"].value_counts(sort=False).sort_index().items()
    }
    source_counts = {
        str(source): int(count)
        for source, count in probe["source"]
        .astype(str)
        .value_counts(sort=False)
        .sort_index()
        .items()
    }
    return {
        "split": "train",
        "n_rows": len(probe),
        "label_counts": label_counts,
        "source_counts": source_counts,
        "source_metadata_preserved": True,
        "sampling": "deterministic balanced sampling from saved task_splits['train'] only",
    }


def _similarity_settings(params: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the original CKA settings recorded in the sensitivity summary.

    Args:
        params: Saved similarity-stage parameter mapping.

    Returns:
        JSON-compatible subset of pooling, token, batch, DBS, and threshold settings.

    Raises:
        KeyError: If a required similarity parameter is absent.
    """
    return {
        "pooling": str(params["pooling"]),
        "max_seq_len": int(params["max_seq_len"]),
        "cka_batch_size": int(params["cka"]["batch_size"]),
        "dbs_box": int(params["dbs"]["box"]),
        "r1_quantile": float(params["thresholds"]["r1_quantile"]),
        "r2_quantile": float(params["thresholds"]["r2_quantile"]),
    }


def _corrected_dbs(
    matrices: Mapping[str, Any],
    *,
    surrogate_names: Sequence[str],
    box: int,
) -> dict[str, float]:
    """Recompute Bresenham diagonal-box DBS from saved CKA matrices.

    Args:
        matrices: Saved target-vs-surrogate CKA matrices by surrogate.
        surrogate_names: Exact surrogate names whose corrected DBS values are needed.
        box: Diagonal-box half-width.

    Returns:
        Corrected DBS mapping for the requested surrogates.

    Raises:
        ValueError: If a requested surrogate has no saved CKA matrix.
    """
    missing = sorted(set(surrogate_names).difference(matrices))
    if missing:
        msg = f"saved original CKA matrices are missing surrogates: {missing}"
        raise ValueError(msg)
    return {
        name: diagonal_box_similarity(np.asarray(matrices[name], dtype=np.float64), box)
        for name in surrogate_names
    }


def _mapping_field(mapping: Mapping[str, Any], field: str, *, context: str) -> Mapping[str, Any]:
    """Return a nested mapping field or raise a context-specific error.

    Args:
        mapping: Parent mapping.
        field: Required child field name.
        context: Parent context for error messages.

    Returns:
        Nested mapping value.

    Raises:
        ValueError: If the field is absent or not a mapping.
    """
    value = mapping.get(field)
    if not isinstance(value, Mapping):
        msg = f"{context} must contain mapping field {field!r}"
        raise ValueError(msg)
    return value


def _true_flip_rates(
    target_audit_summary: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract defined macro true-target-flip mean and maximum values by surrogate.

    Args:
        target_audit_summary: Final saved target-outcome audit summary.

    Returns:
        Tuple of ``(macro_mean, macro_max)`` mappings keyed by surrogate.

    Raises:
        ValueError: If the final full-cohort macro structure is invalid or has duplicate names.
    """
    full_cohort = _mapping_field(
        target_audit_summary, "full_cohort", context="target audit summary"
    )
    surrogate_macro = _mapping_field(full_cohort, "surrogate_macro", context="full_cohort")
    rows = surrogate_macro.get("rows")
    if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Sequence):
        msg = "full_cohort.surrogate_macro.rows must be a sequence"
        raise ValueError(msg)
    means: dict[str, float] = {}
    maxima: dict[str, float] = {}
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            msg = "full_cohort.surrogate_macro.rows must contain mappings"
            raise ValueError(msg)
        name = str(raw_row.get("surrogate", ""))
        if not name:
            msg = "full_cohort.surrogate_macro.rows contains a blank surrogate name"
            raise ValueError(msg)
        if name in means or name in maxima:
            msg = f"full_cohort.surrogate_macro.rows contains duplicate surrogate {name!r}"
            raise ValueError(msg)
        mean = _optional_finite_float(
            raw_row.get("true_target_flip_rate_macro_mean"),
            field=f"full-cohort macro true-target-flip mean for {name!r}",
        )
        maximum = _optional_finite_float(
            raw_row.get("true_target_flip_rate_macro_max"),
            field=f"full-cohort macro true-target-flip maximum for {name!r}",
        )
        if mean is not None:
            means[name] = mean
        if maximum is not None:
            maxima[name] = maximum
    return means, maxima


def _true_flip_association(
    training_mean_cka: Mapping[str, float],
    true_flip_mean: Mapping[str, float],
) -> dict[str, Any]:
    """Associate training-probe mean CKA with saved full-cohort true-target-flip rates.

    Args:
        training_mean_cka: Training-probe mean CKA by surrogate.
        true_flip_mean: Defined full-cohort macro true-target-flip rates by surrogate.

    Returns:
        Labelled post-hoc Spearman association for the shared surrogate pool.
    """
    names = sorted(set(training_mean_cka).intersection(true_flip_mean))
    return _association_block(
        [training_mean_cka[name] for name in names],
        [true_flip_mean[name] for name in names],
        pool_label="surrogates with defined full-cohort macro true-target-flip rates",
    )


def _ablation_rng(seed: int) -> np.random.Generator:
    """Derive the deterministic RNG reserved for training-probe selection ablation.

    Args:
        seed: Root reproducibility seed.

    Returns:
        Seeded NumPy generator.
    """
    seed_sequence = np.random.SeedSequence(
        [derive_seeds(seed).numpy, _TRUE_FLIP_ABLATION_COMPONENT]
    )
    return np.random.default_rng(seed_sequence)


def _true_flip_ablation(
    selection: Mapping[str, Sequence[str]],
    true_flip_mean: Mapping[str, float],
    true_flip_max: Mapping[str, float],
    *,
    risk_params: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Run the post-hoc true-target-flip M1/M2 ablation for training-probe membership.

    Args:
        selection: M1/M2 membership calibrated from training-probe mean CKA.
        true_flip_mean: Defined full-cohort macro mean true-target-flip rates.
        true_flip_max: Defined full-cohort macro maximum true-target-flip rates.
        risk_params: Risk-stage ablation parameters.
        seed: Root reproducibility seed.

    Returns:
        Labelled exact-or-Monte-Carlo M1/M2 ablation metadata.
    """
    usable_names = set(true_flip_mean).intersection(true_flip_max)
    m1 = [name for name in selection["M1"] if name in usable_names]
    m2 = [name for name in selection["M2"] if name in usable_names]
    designed_pool = {
        "label": "training-probe M1/M2 designed pool with defined full-cohort outcomes",
        "unit": "one row per surrogate",
        "n_m1": len(m1),
        "n_m2": len(m2),
    }
    if not m1 or not m2:
        return {
            "designed_pool": designed_pool,
            "one_sided_exchangeability_null": {
                "status": "not_estimable",
                "reason": (
                    "no defined full-cohort true-target-flip outcomes for one selection group"
                ),
                "exact": False,
                "mean_p_value": None,
                "max_p_value": None,
                "p_value_label": "not estimable for the training-probe M1/M2 designed pool",
            },
        }
    n_permutations = int(risk_params["ablation"]["n_permutations"])
    stats = selection_ablation(
        true_flip_mean,
        true_flip_max,
        m1,
        m2,
        n_permutations=n_permutations,
        rng=_ablation_rng(seed),
    )
    exact = bool(stats["exact"])
    p_value_label = (
        "exact enumeration under the exchangeability null for the training-probe M1/M2 "
        "designed pool"
        if exact
        else (
            "Monte Carlo label permutation under the exchangeability null for the "
            "training-probe M1/M2 designed pool"
        )
    )
    return {
        "designed_pool": designed_pool,
        "one_sided_exchangeability_null": {
            "status": "estimated",
            **stats,
            "p_value_label": p_value_label,
        },
    }


def build_training_probe_sensitivity_summary(  # noqa: PLR0913
    *,
    training_probe: pd.DataFrame,
    training_similarity: pd.DataFrame,
    training_thresholds: Mapping[str, Any],
    training_selection: Mapping[str, Any],
    original_similarity: pd.DataFrame,
    original_selection: Mapping[str, Any],
    original_cka_matrices: Mapping[str, Any],
    target_audit_summary: Mapping[str, Any],
    similarity_params: Mapping[str, Any],
    risk_params: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Summarize training-only CKA sensitivity without generating new attack evidence.

    Args:
        training_probe: Deterministically sampled training-only probe.
        training_similarity: CKA/DBS results from the training-only probe.
        training_thresholds: Training-probe calibrated thresholds.
        training_selection: Training-probe M1/M2 membership.
        original_similarity: Saved original CKA table; only its mean CKA is used.
        original_selection: Saved original M1/M2 membership.
        original_cka_matrices: Saved original CKA matrices for corrected DBS recomputation.
        target_audit_summary: Final stable target-outcome audit aggregate.
        similarity_params: Original CKA settings, including Bresenham DBS box width.
        risk_params: Risk-stage ablation parameters.
        seed: Root reproducibility seed.

    Returns:
        Strict JSON-compatible post-hoc sensitivity summary.

    Raises:
        ValueError: If similarity inputs, memberships, matrices, or target-audit macro rows are
            malformed or incompatible.
    """
    original_mean_cka = _metric_by_surrogate(
        original_similarity,
        metric="mean_cka",
        artifact="original similarity table",
    )
    training_mean_cka = _metric_by_surrogate(
        training_similarity,
        metric="mean_cka",
        artifact="training-probe similarity table",
    )
    original_members = _selection_members(
        original_selection,
        artifact="original surrogate selection",
    )
    training_members = _selection_members(
        training_selection,
        artifact="training-probe surrogate selection",
    )
    _validate_selection_pool(
        original_members,
        surrogate_names=set(original_mean_cka),
        artifact="original surrogate selection",
    )
    _validate_selection_pool(
        training_members,
        surrogate_names=set(training_mean_cka),
        artifact="training-probe surrogate selection",
    )

    settings = _similarity_settings(similarity_params)
    original_dbs = _corrected_dbs(
        original_cka_matrices,
        surrogate_names=sorted(original_mean_cka),
        box=int(settings["dbs_box"]),
    )
    training_dbs = _metric_by_surrogate(
        training_similarity,
        metric="dbs",
        artifact="training-probe similarity table",
    )

    _, original_mean_values, training_mean_values = _aligned_values(
        original_mean_cka,
        training_mean_cka,
        first_name="original mean CKA",
        second_name="training-probe mean CKA",
    )
    _, original_dbs_values, training_dbs_values = _aligned_values(
        original_dbs,
        training_dbs,
        first_name="original corrected DBS",
        second_name="training-probe corrected DBS",
    )
    true_flip_mean, true_flip_max = _true_flip_rates(target_audit_summary)

    return {
        "interpretation": {
            "analysis_label": (
                "post-hoc training-only CKA sensitivity analysis; not a replacement for the "
                "original CKA analysis or independent validation"
            ),
            "probe_scope": (
                "The representation probe uses only rows sampled from the existing local "
                "training split, which local fine-tuned surrogates already used."
            ),
            "attack_data_note": (
                "The outcome comparison reuses final target_audit_summary aggregates; no new "
                "attack data was created."
            ),
            "inference_note": (
                "No new surrogate models were added and no M1/M2 sample size increased."
            ),
        },
        "probe_metadata": _probe_metadata(training_probe),
        "similarity_settings": settings,
        "rank_stability": {
            "original_vs_training_mean_cka": _association_block(
                original_mean_values,
                training_mean_values,
                pool_label="original and training-probe CKA configured surrogate pool",
            ),
            "original_vs_training_corrected_dbs": _association_block(
                original_dbs_values,
                training_dbs_values,
                pool_label="original and training-probe corrected-DBS configured surrogate pool",
            ),
        },
        "selection_membership": {
            "original": original_members,
            "training_probe": {
                **training_members,
                "r1": _finite_float(training_thresholds["r1"], field="training-probe r1"),
                "r2": _finite_float(training_thresholds["r2"], field="training-probe r2"),
                "signal": "mean_cka",
            },
        },
        "membership_overlap": _membership_overlap(original_members, training_members),
        "true_target_flip_sensitivity": {
            "analysis_label": (
                "post-hoc full-cohort true-target-flip outcome sensitivity using "
                "training-probe CKA and training-probe M1/M2 membership"
            ),
            "outcome_source": (
                "final target_audit_summary full_cohort surrogate macro true-target-flip rates"
            ),
            "training_probe_mean_cka_vs_full_cohort_macro_true_target_flip_rate": (
                _true_flip_association(training_mean_cka, true_flip_mean)
            ),
            "training_probe_selection_ablation": _true_flip_ablation(
                training_members,
                true_flip_mean,
                true_flip_max,
                risk_params=risk_params,
                seed=seed,
            ),
        },
    }
