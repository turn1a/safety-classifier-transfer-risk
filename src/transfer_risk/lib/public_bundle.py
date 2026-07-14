"""Pure helpers for the public reporting artifact bundle.

These functions transform saved pipeline tables and metadata into the recruiter-facing
CSV/JSON bundle under ``docs/artifacts/``. They perform no I/O and never load models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from transfer_risk.lib.association import spearman_association
from transfer_risk.lib.dbs import diagonal_box_similarity

FloatArray = npt.NDArray[np.float64]
_PUBLIC_MASTER_COLUMNS = [
    "surrogate",
    "recipe",
    "n_successful",
    "n_target_benign",
    "transfer_rate",
    "mean_cka",
    "dbs",
    "target",
]
_SURROGATE_SUMMARY_COLUMNS = [
    "surrogate",
    "mean_cka",
    "dbs",
    "macro_mean_transfer",
    "macro_min_transfer",
    "macro_max_transfer",
    "total_successful",
    "total_target_benign",
    "pooled_target_benign_rate",
    "similarity_group",
]
_MIN_CORR = 3
_ATTACK_RESULT_TYPES = {
    "SuccessfulAttackResult": "successful",
    "FailedAttackResult": "failed",
    "SkippedAttackResult": "skipped",
}
_REDACTED_QUALITATIVE_AUDITS: tuple[dict[str, Any], ...] = (
    {
        "surrogate": "deberta-base-ft-seed",
        "recipe": "bae",
        "n_words_changed": 1,
        "label": "semantic_preservation_uncertain",
        "change_summary": "A one-token lexical substitution was reviewed without retaining text.",
        "audit_note": (
            "Historical conditional target-benign audit only; no audited baseline outcome is "
            "reported here."
        ),
    },
    {
        "surrogate": "bert-base-ft",
        "recipe": "bae",
        "n_words_changed": 1,
        "label": "meaning_changed",
        "change_summary": "The assessed lexical substitution materially changed meaning.",
        "audit_note": (
            "Historical conditional target-benign audit only; no audited baseline outcome is "
            "reported here."
        ),
    },
)
_QUALITATIVE_AUDIT_FIELDS = frozenset(
    {"surrogate", "recipe", "n_words_changed", "label", "change_summary", "audit_note"}
)
_FORBIDDEN_QUALITATIVE_AUDIT_FIELDS = frozenset(
    {"original", "perturbed", "text", "prompt", "source", "user"}
)


def reconstruct_n_target_benign(
    n_successful: pd.Series | int,
    transfer_rate: pd.Series | float,
) -> pd.Series | int:
    """Reconstruct integer target-benign counts from rate and denominator.

    Args:
        n_successful: Successful surrogate-attack counts per cell.
        transfer_rate: Conditional target-benign rate per cell.

    Returns:
        Rounded integer target-benign counts with the same shape as the inputs.
    """
    product = pd.Series(n_successful) * pd.Series(transfer_rate)
    rounded = product.round().astype(int)
    if isinstance(n_successful, int):
        return int(rounded.iloc[0])
    return rounded


def corrected_dbs_by_surrogate(
    cka_matrices: dict[str, FloatArray],
    *,
    box: int,
) -> dict[str, float]:
    """Compute corrected DBS for every surrogate from saved CKA matrices.

    Args:
        cka_matrices: Mapping ``surrogate -> layer-by-layer CKA matrix``.
        box: Diagonal-box half-width passed to :func:`diagonal_box_similarity`.

    Returns:
        Mapping ``surrogate -> corrected DBS``.
    """
    return {
        name: diagonal_box_similarity(np.asarray(matrix, dtype=np.float64), box)
        for name, matrix in cka_matrices.items()
    }


def apply_corrected_dbs(
    master: pd.DataFrame,
    cka_matrices: dict[str, FloatArray],
    *,
    box: int,
) -> pd.DataFrame:
    """Return a copy of the master table with corrected DBS and target-benign counts.

    Args:
        master: Saved master results table (one row per surrogate x recipe).
        cka_matrices: Saved CKA matrices keyed by surrogate name.
        box: Diagonal-box half-width for DBS recomputation.

    Returns:
        Copy of ``master`` with overwritten ``dbs`` and added ``n_target_benign``.
    """
    corrected = corrected_dbs_by_surrogate(cka_matrices, box=box)
    updated = master.copy()
    updated["dbs"] = updated["surrogate"].map(corrected)
    updated["n_target_benign"] = reconstruct_n_target_benign(
        updated["n_successful"],
        updated["transfer_rate"],
    )
    return updated


def assign_similarity_group(surrogate: str, selection: dict[str, Any]) -> str:
    """Map a surrogate to ``M1``, ``M2``, or ``middle`` using saved selection bands.

    Args:
        surrogate: Surrogate name.
        selection: Saved ``surrogate_selection`` dict with ``M1`` and ``M2`` lists.

    Returns:
        ``"M1"``, ``"M2"``, or ``"middle"``.
    """
    if surrogate in selection.get("M1", []):
        return "M1"
    if surrogate in selection.get("M2", []):
        return "M2"
    return "middle"


def build_surrogate_summary(
    master: pd.DataFrame,
    selection: dict[str, Any],
) -> pd.DataFrame:
    """Aggregate corrected master rows to one summary row per surrogate.

    Args:
        master: Master table with corrected ``dbs`` and ``n_target_benign``.
        selection: Saved ``surrogate_selection`` dict.

    Returns:
        Ten-row (for the production run) surrogate summary dataframe.
    """
    working = master.copy()
    if "n_target_benign" not in working.columns:
        working["n_target_benign"] = reconstruct_n_target_benign(
            working["n_successful"],
            working["transfer_rate"],
        )
    grouped = working.groupby("surrogate", as_index=False).agg(
        mean_cka=("mean_cka", "first"),
        dbs=("dbs", "first"),
        macro_mean_transfer=("transfer_rate", "mean"),
        macro_min_transfer=("transfer_rate", "min"),
        macro_max_transfer=("transfer_rate", "max"),
        total_successful=("n_successful", "sum"),
        total_target_benign=("n_target_benign", "sum"),
    )
    grouped["pooled_target_benign_rate"] = (
        grouped["total_target_benign"] / grouped["total_successful"]
    )
    grouped["similarity_group"] = grouped["surrogate"].map(
        lambda name: assign_similarity_group(str(name), selection)
    )
    return grouped[_SURROGATE_SUMMARY_COLUMNS].sort_values("mean_cka").reset_index(drop=True)


def export_public_master_table(master: pd.DataFrame) -> pd.DataFrame:
    """Select and order the public master-results columns.

    Args:
        master: Corrected master table including ``n_target_benign``.

    Returns:
        Public master table ready for CSV export.
    """
    return master[_PUBLIC_MASTER_COLUMNS].copy()


def _association_by_feature(
    table: pd.DataFrame,
    features: tuple[str, ...] = ("mean_cka", "dbs"),
) -> dict[str, dict[str, float | int | bool]]:
    """Compute Spearman association for each feature against transfer rate.

    Args:
        table: Input table with feature columns and ``transfer_rate``.
        features: Feature column names to correlate.

    Returns:
        Mapping ``feature -> {rho, two_sided_p, n, exact}``.
    """
    if len(table) < _MIN_CORR:
        return {}
    association: dict[str, dict[str, float | int | bool]] = {}
    for feature in features:
        association[feature] = spearman_association(
            table[feature].tolist(),
            table["transfer_rate"].tolist(),
        )
    return association


def build_public_run_metrics(
    master: pd.DataFrame,
    ablation: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Build the corrected public run-metrics document.

    Args:
        master: Corrected master table.
        ablation: Saved ablation result dict.
        thresholds: Calibrated ``{"r1", "r2"}`` thresholds.

    Returns:
        Nested public metrics dict for ``docs/artifacts/run_metrics.json``.
    """
    surrogate_table = (
        master.groupby("surrogate", as_index=False)
        .agg(
            mean_cka=("mean_cka", "first"),
            dbs=("dbs", "first"),
            transfer_rate=("transfer_rate", "mean"),
        )
        .sort_values("mean_cka")
    )
    recipe_sensitivity: dict[str, dict[str, float | int | bool]] = {}
    for recipe, recipe_frame in master.groupby("recipe"):
        stats = _association_by_feature(recipe_frame, features=("mean_cka",))
        if "mean_cka" in stats:
            recipe_sensitivity[str(recipe)] = stats["mean_cka"]
    successful = master["n_successful"]
    return {
        "counts": {
            "n_cells": len(master),
            "n_surrogates": int(master["surrogate"].nunique()),
            "n_recipes": int(master["recipe"].nunique()),
        },
        "transfer": {
            "macro_cell_mean": float(master["transfer_rate"].mean()),
            "pooled_target_benign": int(master["n_target_benign"].sum()),
            "pooled_successful": int(successful.sum()),
            "pooled_target_benign_rate": float(master["n_target_benign"].sum() / successful.sum()),
            "cell_successful_min": int(successful.min()),
            "cell_successful_median": float(successful.median()),
            "cell_successful_mean": float(successful.mean()),
            "cell_successful_max": int(successful.max()),
        },
        "associations": {
            "surrogate": _association_by_feature(surrogate_table),
            "recipe_sensitivity": recipe_sensitivity,
        },
        "thresholds": {"r1": float(thresholds["r1"]), "r2": float(thresholds["r2"])},
        "ablation": {
            "m1_mean": float(ablation["m1_mean"]),
            "m2_mean": float(ablation["m2_mean"]),
            "mean_diff_pp": float(ablation["mean_diff_pp"]),
            "mean_p_value": float(ablation["mean_p_value"]),
            "max_diff_pp": float(ablation.get("max_diff_pp", 0.0)),
            "max_p_value": float(ablation.get("max_p_value", 1.0)),
            "n_m1": int(ablation.get("n_m1", len(ablation.get("m1", [])))),
            "n_m2": int(ablation.get("n_m2", len(ablation.get("m2", [])))),
        },
    }


def summarize_attack_outcomes(
    adversarial_examples: dict[str, list[dict[str, Any]]],
) -> dict[str, int | float]:
    """Aggregate attack-result classes without retaining adversarial text.

    Args:
        adversarial_examples: Catalog-wired mapping from attack cell to result records.
            Only each record's ``result_type`` is inspected.

    Returns:
        Counts for attempted, eligible, successful, failed, and skipped attacks, plus
        successful attacks divided by eligible attacks.

    Raises:
        ValueError: If a record has an unknown or missing ``result_type``.
    """
    counts = {"successful": 0, "failed": 0, "skipped": 0}
    for records in adversarial_examples.values():
        for record in records:
            result_type = str(record.get("result_type", ""))
            outcome = _ATTACK_RESULT_TYPES.get(result_type)
            if outcome is None:
                msg = f"Unknown attack result type: {result_type or '<missing>'}"
                raise ValueError(msg)
            counts[outcome] += 1
    eligible = counts["successful"] + counts["failed"]
    attempted = eligible + counts["skipped"]
    success_rate = counts["successful"] / eligible if eligible else 0.0
    return {
        "attempted": attempted,
        "eligible": eligible,
        "successful": counts["successful"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "eligible_attack_success_rate": success_rate,
    }


def build_results_manifest(manifest_inputs: dict[str, Any]) -> dict[str, Any]:
    """Build the safe public results manifest.

    Args:
        manifest_inputs: Flat manifest inputs assembled by the reporting node.

    Returns:
        Manifest dict safe for public release.
    """
    return {
        "schema_version": str(manifest_inputs["schema_version"]),
        "experiment_commit": str(manifest_inputs["experiment_commit"]),
        "root_seed": int(manifest_inputs["root_seed"]),
        "uv_lock_sha256": str(manifest_inputs["uv_lock_sha256"]),
        "target": str(manifest_inputs["target"]),
        "probe_window": dict(manifest_inputs["probe_window"]),
        "attack_window": dict(manifest_inputs["attack_window"]),
        "transfer_window": dict(manifest_inputs["transfer_window"]),
        "attack_attempts_per_cell": int(manifest_inputs["attack_attempts_per_cell"]),
        "query_budget": int(manifest_inputs["query_budget"]),
        "n_surrogates": int(manifest_inputs["n_surrogates"]),
        "n_recipes": int(manifest_inputs["n_recipes"]),
        "hardware": dict(manifest_inputs["hardware"]),
        "completion_note": str(manifest_inputs["completion_note"]),
    }


def validate_redacted_qualitative_audit(records: list[dict[str, Any]]) -> None:
    """Validate that qualitative audit records exactly match the safe public schema.

    Args:
        records: Candidate public qualitative audit records.

    Returns:
        ``None`` after validating the fixed safe release schema.

    Raises:
        ValueError: If records expose source-text fields or differ from the fixed safe audit.
    """
    expected = [dict(record) for record in _REDACTED_QUALITATIVE_AUDITS]
    for record in records:
        fields = set(record)
        forbidden = fields.intersection(_FORBIDDEN_QUALITATIVE_AUDIT_FIELDS)
        if forbidden:
            msg = "redacted qualitative audit contains forbidden source-text fields"
            raise ValueError(msg)
        if fields != _QUALITATIVE_AUDIT_FIELDS:
            msg = "redacted qualitative audit fields must match the safe public schema"
            raise ValueError(msg)
        if not isinstance(record["n_words_changed"], int):
            msg = "redacted qualitative audit n_words_changed must be an integer"
            raise ValueError(msg)
        if any(not isinstance(record[field], str) for field in fields - {"n_words_changed"}):
            msg = "redacted qualitative audit string fields must be strings"
            raise ValueError(msg)
    if len(records) != len(expected):
        msg = "redacted qualitative audit has an unexpected record count"
        raise ValueError(msg)
    if records != expected:
        msg = "redacted qualitative audit records must match the fixed safe audit"
        raise ValueError(msg)


def build_redacted_qualitative_audit() -> list[dict[str, Any]]:
    """Build the fixed qualitative audit that cannot expose source prompt text.

    Returns:
        A fresh copy of the safe redacted qualitative audit records.
    """
    audit = [dict(record) for record in _REDACTED_QUALITATIVE_AUDITS]
    validate_redacted_qualitative_audit(audit)
    return audit


def build_model_validation_summary(
    registry: dict[str, Any],
    metadata_by_surrogate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate model validation metadata for public release.

    Args:
        registry: Saved ``surrogate_registry`` dict.
        metadata_by_surrogate: Mapping ``surrogate -> metadata fragment``.

    Returns:
        Summary dict with ``target`` and per-model ``kind`` / validation fields.
    """
    models: list[dict[str, Any]] = []
    for spec in registry.get("surrogates", []):
        name = str(spec["name"])
        entry: dict[str, Any] = {"name": name, "kind": spec.get("kind")}
        meta = metadata_by_surrogate.get(name, {})
        for field in ("val_accuracy", "num_params"):
            if field in meta:
                entry[field] = meta[field]
        models.append(entry)
    return {"target": registry.get("target"), "models": models}


def publish_dataset_audit(dataset_audit: dict[str, Any]) -> dict[str, Any]:
    """Return the safe dataset audit aggregate with per-source count scope.

    Args:
        dataset_audit: Saved dataset audit JSON.

    Returns:
        A copy of the aggregate audit with metadata clarifying that ``per_source``
        counts are post-deduplication.
    """
    published = dict(dataset_audit)
    published["per_source_count_stage"] = "post_deduplication"
    return published
