"""Nodes for the transfer pipeline (SPEC.md §3.1 step 5)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from transfer_risk.devices import resolve_device
from transfer_risk.lib.transfer import transfer_rate
from transfer_risk.modeling import predict

logger = logging.getLogger(__name__)

# The class id meaning "not an injection"; a target prediction equal to it on a perturbed
# prompt means the attack transferred (matches transfer_rate's default).
_BENIGN_LABEL = 0


def assemble_adversarial(
    *cells: list[dict[str, Any]], cellkeys: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """Combine the per-cell adversarial partitions into one ``{cell -> records}`` mapping.

    The cell inputs are wired in ``cellkeys`` order, so this bridges the per-cell
    ``adversarial.{cell}`` partitions (possibly on S3) to the single table the transfer node
    consumes — connecting the attack reduce nodes to the transfer stage in the DAG.
    """
    return dict(zip(cellkeys, cells, strict=True))


def _words_changed(original: str, perturbed: str) -> int:
    """Count whitespace-token edits between two strings (a legibility proxy).

    Args:
        original: The original prompt.
        perturbed: The adversarially perturbed prompt.

    Returns:
        Position-wise differing tokens plus any length difference; a small value flags a
        minimally perturbed, easy-to-read example to feature in the write-up.
    """
    original_tokens = original.split()
    perturbed_tokens = perturbed.split()
    substitutions = sum(
        1 for a, b in zip(original_tokens, perturbed_tokens, strict=False) if a != b
    )
    return substitutions + abs(len(original_tokens) - len(perturbed_tokens))


def _select_transferred(
    candidates: list[dict[str, Any]], max_examples: int, max_per_surrogate: int = 2
) -> list[dict[str, Any]]:
    """Pick a small, varied, minimally perturbed set of transferred examples.

    Sorts by perturbation size (fewest word edits first) for legibility, then greedily takes
    examples under a per-surrogate cap so the set spans surrogates rather than repeating one;
    if that leaves it short, it fills from the remaining ordered candidates.

    Args:
        candidates: All collected transferred examples (each a record dict).
        max_examples: Maximum number to keep.
        max_per_surrogate: Cap per surrogate before filling from the remainder.

    Returns:
        The selected records, deterministically ordered.
    """
    ordered = sorted(
        candidates,
        key=lambda record: (
            record["n_words_changed"],
            record["surrogate"],
            record["recipe"],
            record["original"],
        ),
    )
    chosen: list[dict[str, Any]] = []
    per_surrogate: dict[str, int] = {}
    for record in ordered:
        if len(chosen) >= max_examples:
            break
        if per_surrogate.get(record["surrogate"], 0) >= max_per_surrogate:
            continue
        chosen.append(record)
        per_surrogate[record["surrogate"]] = per_surrogate.get(record["surrogate"], 0) + 1
    if len(chosen) < max_examples:
        chosen_ids = {id(record) for record in chosen}
        for record in ordered:
            if len(chosen) >= max_examples:
                break
            if id(record) not in chosen_ids:
                chosen.append(record)
    return chosen


def evaluate_transfer(
    adversarial_examples: dict[str, Any],
    target: dict[str, Any],
    params: dict[str, Any],
    device_params: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Feed successful adversarial examples to the frozen target; record transfer rate.

    The transfer rate per (surrogate, recipe) is the fraction of the surrogate's
    successful adversarial examples (injection -> benign on the surrogate) that the
    frozen target *also* predicts benign — i.e. the attack transfers. Alongside the rate
    table, a few genuinely transferred examples (surrogate and target both flipped) are
    selected for the write-up's before/after illustration.

    Args:
        adversarial_examples: PartitionedDataset of per-cell loaders (one cell per partition).
        target: the frozen target bundle (``{"model", "tokenizer"}``) from ``target_model``.
        params: the ``transfer`` block.
        device_params: the ``device`` block.

    Returns:
        The per-(surrogate, recipe) transfer-rate table and a small list of worked
        transferred examples (``surrogate``, ``recipe``, ``original``, ``perturbed``,
        ``n_words_changed``).
    """
    device = resolve_device(device_params["policy"])
    model = target["model"].to(device).eval()
    tokenizer = target["tokenizer"]
    max_seq_len = int(params.get("max_seq_len", 256))
    batch_size = int(params["batch_size"])
    max_examples = int(params.get("max_transferred_examples", 8))
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for key, records in adversarial_examples.items():
        # adversarial_examples is the assembled {cell -> records} table; the cell key carries the
        # (surrogate, recipe) identity that the per-cell partition filename encoded.
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
        rate = transfer_rate(target_preds)
        rows.append(
            {
                "surrogate": surrogate,
                "recipe": recipe,
                "n_successful": len(successful),
                "transfer_rate": rate,
            }
        )
        for record, prediction in zip(successful, target_preds, strict=True):
            if prediction == _BENIGN_LABEL:
                candidates.append(
                    {
                        "surrogate": surrogate,
                        "recipe": recipe,
                        "original": record["original"],
                        "perturbed": record["perturbed"],
                        "n_words_changed": _words_changed(record["original"], record["perturbed"]),
                    }
                )
        logger.info(
            "%s/%s: transfer rate %.3f over %d successful", surrogate, recipe, rate, len(successful)
        )
    transferred = _select_transferred(candidates, max_examples)
    logger.info(
        "Selected %d transferred examples from %d candidates", len(transferred), len(candidates)
    )
    return pd.DataFrame(rows), transferred


def assemble_results_table(
    transfer_results: pd.DataFrame,
    similarity_table: pd.DataFrame,
    registry: dict[str, Any],
) -> pd.DataFrame:
    """Join transfer rates with the per-surrogate similarity scalars (mean CKA, DBS)."""
    merged = transfer_results.merge(similarity_table, on="surrogate", how="left")
    merged["target"] = registry["target"]
    return merged
