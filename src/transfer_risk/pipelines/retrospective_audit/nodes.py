"""Evaluate cached classifiers without generating attacks or training models."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
import torch

from transfer_risk.devices import resolve_device
from transfer_risk.lib.baseline_audit import summarize_detector_baseline
from transfer_risk.lib.policy_coverage import summarize_original_policy_coverage
from transfer_risk.lib.retrospective_audit import (
    build_retrospective_cells,
    reference_cells_digest,
    verify_reference_counts,
)
from transfer_risk.lib.source_baseline_check import verify_source_baseline_counts
from transfer_risk.modeling import predict

logger = logging.getLogger(__name__)
_CHUNK_SIZE = 256


def build_source_baseline_check(
    linkage: pd.DataFrame,
    baseline: dict[str, Any],
    manifest: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Publish source-specific eligibility parity using existing aggregate baseline results."""
    return verify_source_baseline_counts(
        linkage,
        baseline,
        recipes=params["recipes"],
        attempts_per_cell=int(manifest["attack_attempts_per_cell"]),
    )


def build_original_coverage(
    predictions: pd.DataFrame,
    linkage: pd.DataFrame,
    similarity: pd.DataFrame,
    historical_cells: pd.DataFrame,
    manifest: dict[str, Any],
    attack_params: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Reduce distinct-original coverage after rechecking historical cell counts."""
    excluded = params["targets"]["protectai-v2"]["excluded_surrogates"]
    current, _, _ = build_retrospective_cells(
        predictions, linkage, target_id="protectai-v2", excluded_surrogates=excluded
    )
    receipt = verify_reference_counts(
        current, historical_cells, manifest, params["targets"]["protectai-v2"]
    )
    report = summarize_original_policy_coverage(
        predictions,
        linkage,
        similarity,
        attempts_per_cell=int(manifest["attack_attempts_per_cell"]),
        recipes=attack_params["recipes"],
        excluded_surrogates=excluded,
    )
    report["reference_check"] = receipt
    return report


def fingerprint_loaded_classifier(model: Any, tokenizer: Any) -> dict[str, Any]:
    """Hash current inference weights and tokenizer vocabulary without publishing paths.

    This identifies the loaded local audit. It cannot recover fingerprints missing
    from the original model training or cloud attack execution.

    Args:
        model: Loaded classifier before transfer to the inference device.
        tokenizer: Matching loaded tokenizer.

    Returns:
        SHA-256 digests and the Hub revision recorded by the loaded model, if present.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    vocabulary = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=True)
    config = getattr(model, "config", None)
    return {
        "inference_weights_sha256": digest.hexdigest(),
        "tokenizer_vocab_sha256": hashlib.sha256(vocabulary.encode()).hexdigest(),
        "loaded_hub_revision": getattr(config, "_commit_hash", None),
        "inference_dtype": "float32",
        "provenance_scope": "current_local_audit_only",
    }


def _local_predict(
    bundle: dict[str, Any],
    texts: list[str],
    params: dict[str, Any],
    device_params: dict[str, Any],
    target_id: str,
    predict_fn: Callable[..., list[int]],
) -> list[int]:
    """Score deduplicated strings serially and log counts without exposing text."""
    device = resolve_device(device_params["policy"])
    model = bundle["model"].to(device).eval()
    if hasattr(model, "config"):
        model.config.output_hidden_states = False
    predictions: list[int] = []
    for start in range(0, len(texts), _CHUNK_SIZE):
        chunk = texts[start : start + _CHUNK_SIZE]
        batch = predict_fn(
            model,
            bundle["tokenizer"],
            chunk,
            max_seq_len=int(params["max_seq_len"]),
            batch_size=int(params["batch_size"]),
            device=device,
        )
        if len(batch) != len(chunk):
            msg = "local classifier did not return one prediction per input"
            raise ValueError(msg)
        predictions.extend(batch)
        logger.info(
            "Local classifier audit %s: %d/%d texts", target_id, len(predictions), len(texts)
        )
    model.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()
    return predictions


def run_retrospective_target(
    pairs: pd.DataFrame,
    linkage: pd.DataFrame,
    target: dict[str, Any],
    params: dict[str, Any],
    device_params: dict[str, Any],
    *,
    target_id: str,
    predict_fn: Callable[..., list[int]] = predict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Audit every retained archived pair on one catalog-loaded classifier.

    Args:
        pairs: Private exact original/perturbed pairs with stable private identifiers.
        linkage: Private per-source/cell trial counts, including unsuccessful source searches.
        target: Catalog-loaded classifier and tokenizer.
        params: Input windows, batch size, and per-target label/exclusion definitions.
        device_params: Local device policy.
        target_id: Configured public identifier.
        predict_fn: Injectable classifier prediction function for tests.

    Returns:
        Private pair predictions, staged aggregate cells/source rows, and a non-text context.
    """
    definition = params["targets"][target_id]
    excluded = definition["excluded_surrogates"]
    retained = linkage.loc[~linkage["surrogate"].isin(excluded)]
    selected = pairs.loc[pairs["pair_id"].isin(retained["pair_id"])].copy()
    fingerprint = fingerprint_loaded_classifier(target["model"], target["tokenizer"])
    texts = sorted(set(selected["original"]) | set(selected["perturbed"]))
    labels = _local_predict(target, texts, params, device_params, target_id, predict_fn)
    by_text = dict(zip(texts, labels, strict=True))
    predictions = selected[["pair_id", "original_id"]].copy()
    predictions["original_pred"] = selected["original"].map(by_text)
    predictions["perturbed_pred"] = selected["perturbed"].map(by_text)
    cells, sources, context = build_retrospective_cells(
        predictions,
        linkage,
        target_id=target_id,
        positive_label=int(definition["positive_label"]),
        benign_label=int(definition["benign_label"]),
        excluded_surrogates=excluded,
    )
    context.update(fingerprint)
    context.update(
        {
            "repo_id": definition["repo_id"],
            "positive_label_name": definition["positive_label_name"],
            "positive_label": int(definition["positive_label"]),
            "benign_label": int(definition["benign_label"]),
            "unique_inferred_texts": len(texts),
            "max_seq_len": int(params["max_seq_len"]),
            "device": str(resolve_device(device_params["policy"])),
            "new_training": False,
            "new_attack_searches": False,
        }
    )
    return predictions.reset_index(drop=True), cells, sources, context


def publish_retrospective_tables(
    receipt: dict[str, Any],
    reference: pd.DataFrame,
    cells: pd.DataFrame,
    sources: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Release aggregate tables only after a matching staged-reference check succeeds."""
    if receipt.get("counts_agree") is not True:
        msg = "candidate publication requires a successful reference check"
        raise ValueError(msg)
    if receipt.get("reference_cells_sha256") != reference_cells_digest(reference):
        msg = "candidate publication rejected a stale reference receipt"
        raise ValueError(msg)
    return cells, sources


def run_detector_baseline(
    task_splits: dict[str, pd.DataFrame],
    target: dict[str, Any],
    params: dict[str, Any],
    device_params: dict[str, Any],
    *,
    target_id: str,
    predict_fn: Callable[..., list[int]] = predict,
) -> dict[str, Any]:
    """Measure canonical held-out recall and benign false positives on cached weights."""
    test = task_splits["test"]
    texts = test["text"].str.slice(stop=int(params["max_prompt_chars"])).tolist()
    fingerprint = fingerprint_loaded_classifier(target["model"], target["tokenizer"])
    labels = _local_predict(target, texts, params, device_params, target_id, predict_fn)
    summary = summarize_detector_baseline(
        test["label"].astype(int).tolist(), labels, test["source"].tolist(), target_id
    )
    summary.update(fingerprint)
    summary.update(
        {
            "max_seq_len": int(params["max_seq_len"]),
            "max_prompt_chars": int(params["max_prompt_chars"]),
            "device": str(resolve_device(device_params["policy"])),
            "analysis_status": "post_hoc_cached_checkpoint_baseline",
            "independent_external_test": False,
            "new_training": False,
            "new_attack_searches": False,
        }
    )
    return summary


def assemble_audit_reports(*reports: dict[str, Any]) -> dict[str, Any]:
    """Collect catalog-owned classifier reports without exposing private evidence."""
    return {
        "analysis_status": "retrospective_local_classifier_inference",
        "reports": list(reports),
        "scope": "Saved texts and cached model checkpoints; no regenerated attacks or training.",
    }


def assemble_retrospective_reports(
    reference_check: dict[str, Any], *reports: dict[str, Any]
) -> dict[str, Any]:
    """Require the reference receipt and its retained counts before aggregate publication."""
    reference = next((report for report in reports if report["target"] == "protectai-v2"), None)
    if reference_check.get("counts_agree") is not True or reference is None:
        msg = "retrospective publication requires a successful reference check"
        raise ValueError(msg)
    if reference["included_surrogates"] != reference_check["retained_surrogates"] or any(
        reference["aggregate_counts"][column] != value
        for column, value in reference_check["aggregate_counts"].items()
    ):
        msg = "retrospective reference receipt does not match the retained model/count context"
        raise ValueError(msg)
    result = assemble_audit_reports(*reports)
    result["reference_check"] = reference_check
    return result
