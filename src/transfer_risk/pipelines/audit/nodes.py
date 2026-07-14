"""Nodes for the post-hoc target-outcome audit pipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from transfer_risk.devices import resolve_device
from transfer_risk.lib import target_audit
from transfer_risk.modeling import predict

logger = logging.getLogger(__name__)

PredictFn = Callable[..., list[int]]


def _predict_unique_texts(
    texts: list[str],
    *,
    predict_fn: PredictFn,
    model: Any,
    tokenizer: Any,
    max_seq_len: int,
    batch_size: int,
    device: Any,
) -> dict[str, int]:
    """Run target inference on deduplicated texts and return a text->label map."""
    if not texts:
        return {}
    predictions = predict_fn(
        model,
        tokenizer,
        texts,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=device,
    )
    return dict(zip(texts, predictions, strict=True))


def run_target_audit(
    adversarial_examples: dict[str, list[dict[str, Any]]],
    target: dict[str, Any],
    task_splits: dict[str, pd.DataFrame],
    master_results_table: pd.DataFrame,
    transfer_params: dict[str, Any],
    audit_params: dict[str, Any],
    device_params: dict[str, Any],
    *,
    predict_fn: PredictFn = predict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate the frozen target once and persist only raw, prediction-derived aggregates.

    Reuses successful source attack records, deduplicates target inputs, and records
    original/perturbed target predictions to quantify conditional benign rates and true target
    flips. The legacy-shaped raw cell table is finalized separately so corrected DBS can be
    applied without repeating target inference. No surrogate models are loaded.

    Args:
        adversarial_examples: Saved ``{surrogate__recipe: records}`` attack table.
        target: Frozen target bundle from ``target_model``.
        task_splits: Saved train/val/test splits for source attribution.
        master_results_table: Saved master table used only to retain the legacy raw cell grid.
        transfer_params: Transfer-stage parameters (``max_seq_len``, ``batch_size``,
            ``benign_label``).
        audit_params: Audit parameters (``eval_set_size``, ``max_prompt_chars``,
            ``excluded_training_source``).
        device_params: Device resolution parameters.
        predict_fn: Injectable prediction function for tests.

    Returns:
        Legacy-compatible raw cells, raw source aggregates, and non-text raw context.
    """
    benign_label = int(transfer_params.get("benign_label", 0))
    injection_label = int(audit_params.get("injection_label", 1))
    max_seq_len = int(transfer_params.get("max_seq_len", 256))
    batch_size = int(transfer_params["batch_size"])
    eval_set_size = int(audit_params["eval_set_size"])
    max_prompt_chars = int(audit_params["max_prompt_chars"])
    excluded_source = str(
        audit_params.get("excluded_training_source", target_audit.KNOWN_TARGET_TRAINING_SOURCE)
    )
    if excluded_source != target_audit.KNOWN_TARGET_TRAINING_SOURCE:
        msg = (
            f"audit.excluded_training_source must be {target_audit.KNOWN_TARGET_TRAINING_SOURCE!r}"
        )
        raise ValueError(msg)

    source_map = target_audit.build_eval_original_source_map(
        task_splits["test"],
        eval_set_size=eval_set_size,
        max_prompt_chars=max_prompt_chars,
    )
    flattened = target_audit.flatten_successful_records(adversarial_examples)
    annotated = target_audit.assign_sources_to_records(
        flattened,
        source_map,
        max_prompt_chars=max_prompt_chars,
    )
    unique_originals = target_audit.dedupe_texts([record["original"] for record in annotated])
    unique_perturbed = target_audit.dedupe_texts([record["perturbed"] for record in annotated])
    unique_text_stats = target_audit.collect_unique_texts(annotated)

    device = resolve_device(device_params["policy"])
    model = target["model"].to(device).eval()
    tokenizer = target["tokenizer"]
    original_predictions = _predict_unique_texts(
        unique_originals,
        predict_fn=predict_fn,
        model=model,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=device,
    )
    perturbed_predictions = _predict_unique_texts(
        unique_perturbed,
        predict_fn=predict_fn,
        model=model,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=device,
    )
    prediction_records = target_audit.attach_predictions(
        annotated, original_predictions, perturbed_predictions
    )
    baseline_counts = target_audit.baseline_counts_on_unique_originals(
        unique_originals,
        original_predictions,
        benign_label=benign_label,
    )

    cells = target_audit.build_full_grid_cells(
        prediction_records,
        master_results_table,
        benign_label=benign_label,
        injection_label=injection_label,
    )
    sources = target_audit.aggregate_by_source(
        prediction_records,
        benign_label=benign_label,
        injection_label=injection_label,
    )
    context = target_audit.build_raw_audit_context(
        excluded_source=excluded_source,
        unique_text_stats=unique_text_stats,
        baseline_counts=baseline_counts,
    )
    logger.info(
        "Target audit inference aggregates complete: %d source successes, %d true flips",
        int(cells["source_successful"].sum()),
        int(cells["true_target_flips"].sum()),
    )
    return cells, sources, context


def finalize_target_audit(  # noqa: PLR0913
    raw_cells: pd.DataFrame,
    raw_sources: pd.DataFrame,
    raw_context: dict[str, Any],
    master_results_table: pd.DataFrame,
    cka_matrices: dict[str, Any],
    similarity_params: dict[str, Any],
    surrogate_selection: dict[str, Any],
    risk_params: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Finalize audit artifacts with corrected DBS and no target-model access.

    Args:
        raw_cells: Saved raw target-inference cell aggregates.
        raw_sources: Saved raw target-inference source aggregates.
        raw_context: Saved non-text context or a legacy audit summary.
        master_results_table: Saved master result grid with mean CKA values.
        cka_matrices: Saved target-vs-surrogate CKA matrices.
        similarity_params: Similarity parameters supplying the DBS box width.
        surrogate_selection: Saved M1/M2 membership.
        risk_params: Risk parameters for deterministic selection ablations.
        seed: Root reproducibility seed.

    Returns:
        Corrected complete audit cells and final audit summary.
    """
    cells, summary = target_audit.finalize_audit_from_aggregates(
        raw_cells,
        raw_sources,
        raw_context,
        master_results_table,
        cka_matrices,
        similarity_params,
        surrogate_selection,
        risk_params,
        seed,
    )
    logger.info(
        "Finalized target audit with corrected DBS: %d source successes, %d true flips",
        summary["full_cohort"]["source_successful"],
        summary["full_cohort"]["true_target_flips"],
    )
    return cells, summary
