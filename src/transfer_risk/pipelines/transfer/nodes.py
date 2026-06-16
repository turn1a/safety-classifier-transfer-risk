"""Nodes for the transfer pipeline (SPEC.md §3.1 step 5)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from transfer_risk.devices import resolve_device
from transfer_risk.modeling import load_transformer, predict

logger = logging.getLogger(__name__)


def evaluate_transfer(
    adversarial_examples: dict[str, Any],
    registry: dict[str, Any],
    params: dict[str, Any],
    device_params: dict[str, Any],
) -> pd.DataFrame:
    """Feed successful adversarial examples to the frozen target; record transfer rate.

    The transfer rate per (surrogate, recipe) is the fraction of the surrogate's
    successful adversarial examples (injection -> benign on the surrogate) that the
    frozen target *also* predicts benign — i.e. the attack transfers.
    """
    device = resolve_device(device_params["policy"])
    model, tokenizer = load_transformer(registry["target"], device)
    max_seq_len = int(params.get("max_seq_len", 256))
    batch_size = int(params["batch_size"])
    rows: list[dict[str, Any]] = []
    for key, records in adversarial_examples.items():
        surrogate, recipe = key.split("__", 1)
        successful = [record for record in records if record.get("success")]
        if not successful:
            rows.append(
                {"surrogate": surrogate, "recipe": recipe, "n_successful": 0, "transfer_rate": 0.0}
            )
            continue
        perturbed = [record["perturbed"] for record in successful]
        target_preds = predict(
            model,
            tokenizer,
            perturbed,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            device=device,
        )
        transferred = sum(1 for prediction in target_preds if prediction == 0)
        rate = transferred / len(successful)
        rows.append(
            {
                "surrogate": surrogate,
                "recipe": recipe,
                "n_successful": len(successful),
                "transfer_rate": rate,
            }
        )
        logger.info(
            "%s/%s: %d/%d transferred (%.3f)", surrogate, recipe, transferred, len(successful), rate
        )
    return pd.DataFrame(rows)


def assemble_results_table(
    transfer_results: pd.DataFrame,
    similarity_table: pd.DataFrame,
    registry: dict[str, Any],
) -> pd.DataFrame:
    """Join transfer rates with the per-surrogate similarity scalars (mean CKA, DBS)."""
    merged = transfer_results.merge(similarity_table, on="surrogate", how="left")
    merged["target"] = registry["target"]
    return merged
