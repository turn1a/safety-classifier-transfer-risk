"""Load trained classifiers and extract per-layer representations.

Shared glue (imports torch + transformers) used by the similarity stage (and later
the attack/transfer stages). Transformers expose every layer via
``output_hidden_states``; the BiLSTM exposes its two pooled layers via
``representations``. A model "spec" is a manifest entry (or the target) with ``kind``
in {pretrained, finetune, bilstm} and a ``source`` (a HuggingFace id or a local dir).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transfer_risk.pipelines.models.bilstm import BiLSTMClassifier, encode, pad_batch

if TYPE_CHECKING:
    from collections.abc import Mapping

FloatArray = npt.NDArray[np.float64]


def layer_representations(
    spec: Mapping[str, Any],
    texts: list[str],
    *,
    pooling: str,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
) -> list[FloatArray]:
    """Return one pooled ``(n_texts, d)`` matrix per layer for the model in ``spec``."""
    if spec["kind"] == "bilstm":
        return _bilstm_representations(spec["source"], texts, batch_size=batch_size, device=device)
    return _transformer_representations(
        spec["source"],
        texts,
        pooling=pooling,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=device,
    )


def _pool(layer: torch.Tensor, mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool a ``(batch, seq, hidden)`` layer to ``(batch, hidden)``."""
    if pooling == "cls":
        return layer[:, 0, :]
    expanded = mask.unsqueeze(-1).to(layer.dtype)
    summed = (layer * expanded).sum(dim=1)
    return summed / expanded.sum(dim=1).clamp(min=1.0)


def _transformer_representations(
    source: str,
    texts: list[str],
    *,
    pooling: str,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
) -> list[FloatArray]:
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = (
        AutoModelForSequenceClassification.from_pretrained(source, output_hidden_states=True)
        .to(device)
        .eval()
    )
    per_layer: list[list[FloatArray]] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                max_length=max_seq_len,
                padding=True,
                return_tensors="pt",
            ).to(device)
            hidden_states = model(**encoded).hidden_states
            mask = encoded["attention_mask"]
            # Skip hidden_states[0] (the static embedding layer): under CLS pooling its
            # [CLS] vector is identical for every input, so its CKA would be undefined.
            for index, layer in enumerate(hidden_states[1:]):
                pooled: FloatArray = _pool(layer, mask, pooling).cpu().numpy().astype(np.float64)
                if index >= len(per_layer):
                    per_layer.append([])
                per_layer[index].append(pooled)
    return [np.concatenate(chunks, axis=0).astype(np.float64) for chunks in per_layer]


def _bilstm_representations(
    source: str,
    texts: list[str],
    *,
    batch_size: int,
    device: torch.device,
) -> list[FloatArray]:
    config = json.loads((Path(source) / "config.json").read_text())
    vocab = config["vocab"]
    model = BiLSTMClassifier(
        len(vocab),
        config["embed_dim"],
        config["hidden_dim"],
        config["num_layers"],
        config["dropout"],
    )
    model.load_state_dict(torch.load(Path(source) / "model.pt", map_location=device))
    model.to(device).eval()
    max_len = config["max_seq_len"]
    per_layer: list[list[FloatArray]] = [[], []]
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            ids = pad_batch(
                [encode(text, vocab, max_len) for text in texts[start : start + batch_size]]
            ).to(device)
            for index, rep in enumerate(model.representations(ids)):
                per_layer[index].append(rep.cpu().numpy().astype(np.float64))
    return [np.concatenate(chunks, axis=0).astype(np.float64) for chunks in per_layer]
