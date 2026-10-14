"""Compute bounded sample diagnostics locally, with scores hidden from human review."""

from __future__ import annotations

import hashlib
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from transfer_risk.devices import resolve_device
from transfer_risk.lib.semantic_review_tool import build_review_dataset
from transfer_risk.lib.semantic_screens import (
    cosine_pairs,
    entailment_probabilities,
    summarize_semantic_screens,
)
from transfer_risk.pipelines.retrospective_audit.nodes import fingerprint_loaded_classifier


def masked_mean_embeddings(model: Any, encoded: dict[str, Any]) -> npt.NDArray[np.float64]:
    """Pool encoder tokens while excluding padding, without assuming normalized outputs."""
    with torch.inference_mode():
        states = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(states.dtype)
        denominator = mask.sum(dim=1)
        if torch.any(denominator <= 0):
            msg = "cannot pool an empty encoder window"
            raise ValueError(msg)
        pooled = (states * mask).sum(dim=1) / denominator
    return np.asarray(pooled.cpu().numpy(), dtype=float)


def _embedding_batch(
    bundle: dict[str, Any], texts: list[str], params: dict[str, Any], device: torch.device
) -> tuple[npt.NDArray[np.float64], list[int]]:
    """Encode deduplicated strings serially and preserve untruncated token lengths."""
    model, tokenizer = bundle["model"].to(device).eval(), bundle["tokenizer"]
    lengths = tokenizer(texts, truncation=False, return_length=True, verbose=False)["length"]
    vectors = []
    for start in range(0, len(texts), params["batch_size"]):
        encoded = tokenizer(
            texts[start : start + params["batch_size"]],
            padding=True,
            truncation=True,
            max_length=params["encoder_max_tokens"],
            return_tensors="pt",
        ).to(device)
        vectors.append(masked_mean_embeddings(model, encoded))
    model.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()
    return np.concatenate(vectors), [int(value) for value in lengths]


def _nli_batch(
    bundle: dict[str, Any],
    pairs: list[tuple[str, str]],
    params: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, npt.NDArray[np.float64]], list[int]]:
    """Score both premise directions through the explicit NLI class map."""
    model, tokenizer = bundle["model"].to(device).eval(), bundle["tokenizer"]
    lengths, logits = [], []
    for start in range(0, len(pairs), params["batch_size"]):
        batch = pairs[start : start + params["batch_size"]]
        left, right = [pair[0] for pair in batch], [pair[1] for pair in batch]
        lengths.extend(
            tokenizer(left, right, truncation=False, return_length=True, verbose=False)["length"]
        )
        encoded = tokenizer(
            left,
            right,
            padding=True,
            truncation="longest_first",
            max_length=params["nli_max_tokens"],
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits.append(model(**encoded).logits.cpu().numpy())
    model.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()
    mapping = {int(index): label.lower() for index, label in model.config.id2label.items()}
    return entailment_probabilities(np.concatenate(logits), mapping), [int(v) for v in lengths]


def screen_review_sample(
    candidates: pd.DataFrame,
    encoder: dict[str, Any],
    nli: dict[str, Any],
    params: dict[str, Any],
    device_params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Persist private pair scores and publish only sample distributions and coverage."""
    dataset = build_review_dataset(
        cast("list[dict[str, Any]]", candidates.to_dict(orient="records"))
    )
    pairs = dataset["pairs"]
    device = resolve_device(device_params["policy"])
    provenance = {
        "encoder": {
            **encoder["identity"],
            **fingerprint_loaded_classifier(encoder["model"], encoder["tokenizer"]),
        },
        "nli": {**nli["identity"], **fingerprint_loaded_classifier(nli["model"], nli["tokenizer"])},
        "settings": dict(params),
        "pooling": "attention-mask-weighted token mean; cosine normalization",
        "nli_mapping": {str(k): v for k, v in nli["model"].config.id2label.items()},
        "device": device.type,
    }
    texts = list(
        dict.fromkeys(text for pair in pairs for text in (pair["original"], pair["perturbed"]))
    )
    vectors, lengths = _embedding_batch(encoder, texts, params, device)
    positions = {text: index for index, text in enumerate(texts)}
    original_positions = [positions[pair["original"]] for pair in pairs]
    perturbed_positions = [positions[pair["perturbed"]] for pair in pairs]
    text_pairs = [(pair["original"], pair["perturbed"]) for pair in pairs]
    forward, forward_lengths = _nli_batch(nli, text_pairs, params, device)
    backward, backward_lengths = _nli_batch(nli, [(b, a) for a, b in text_pairs], params, device)
    scores = pd.DataFrame(
        {
            "pair_id": [pair["pair_id"] for pair in pairs],
            "original_id": [
                hashlib.sha256(pair["original"].encode()).hexdigest() for pair in pairs
            ],
            "embedding_cosine": cosine_pairs(
                vectors[original_positions], vectors[perturbed_positions]
            ),
            "nli_min_entailment": np.minimum(forward["entailment"], backward["entailment"]),
            "nli_max_contradiction": np.maximum(
                forward["contradiction"], backward["contradiction"]
            ),
            "embedding_original_tokens": [lengths[index] for index in original_positions],
            "embedding_perturbed_tokens": [lengths[index] for index in perturbed_positions],
            "nli_forward_tokens": forward_lengths,
            "nli_backward_tokens": backward_lengths,
            "embedding_truncated": [
                max(lengths[a], lengths[b]) > params["encoder_max_tokens"]
                for a, b in zip(original_positions, perturbed_positions, strict=True)
            ],
            "nli_truncated": [
                max(a, b) > params["nli_max_tokens"]
                for a, b in zip(forward_lengths, backward_lengths, strict=True)
            ],
        }
    )
    for direction, result in (("forward", forward), ("backward", backward)):
        for label, values in result.items():
            scores[f"nli_{direction}_{label}"] = values
    return scores, summarize_semantic_screens(scores, provenance)
