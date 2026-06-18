"""Training procedures for the surrogate pool: transformer fine-tuning + the BiLSTM.

Both use a small explicit torch loop rather than the HF ``Trainer`` — ``model(**enc,
labels=...).loss`` is stable across transformers versions, and the loop keeps seeding
and the MPS device explicit (SPEC.md §7). Each function trains, saves a checkpoint, and
returns manifest metadata (validation accuracy, parameter count).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import torch
from torch.nn.functional import cross_entropy
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from transfer_risk.datasets.surrogate_model_dataset import SurrogateModelDataset
from transfer_risk.pipelines.models.bilstm import BiLSTMClassifier, build_vocab, encode, pad_batch

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

logger = logging.getLogger(__name__)


class _NaNLossError(RuntimeError):
    """Raised when a training step produces a non-finite loss (numerical divergence)."""


def _train_transformer(
    model_id: str,
    tokenizer: Any,
    train_df: pd.DataFrame,
    cfg: dict[str, Any],
    device: torch.device,
    seed: int,
) -> Any:
    """Fine-tune one transformer on ``device`` and return the trained model.

    Uses gradient clipping and a linear warmup (both HF ``Trainer`` defaults) for stability.
    The loss is checked every step; a non-finite value raises :class:`_NaNLossError` at the
    step it appears, so the caller can retry the model on another device without finishing a
    doomed epoch.

    Args:
        model_id: HuggingFace model identifier to load the backbone from.
        tokenizer: tokenizer paired with ``model_id``.
        train_df: training split with ``text`` and ``label`` columns.
        cfg: the ``finetune`` block (``lr``, ``batch_size``, ``epochs``, ``max_seq_len``).
        device: device to train on.
        seed: seed for the weight init and the batch-shuffle generator.

    Returns:
        The fine-tuned model, left on ``device``.

    Raises:
        _NaNLossError: if any training step yields a non-finite loss.
    """
    torch.manual_seed(seed)
    batch_size, max_seq_len = cfg["batch_size"], cfg["max_seq_len"]
    # Force fp32: deberta-v3-base ships a float16 checkpoint, and transformers>=5 loads in
    # the checkpoint dtype by default. Training in raw fp16 without loss scaling overflows to
    # NaN within an epoch (other backbones ship fp32 and were unaffected). fp32 is the stable,
    # device-agnostic choice for a from-scratch fine-tune.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, dtype=torch.float32
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    texts = train_df["text"].tolist()
    labels = train_df["label"].tolist()
    generator = torch.Generator().manual_seed(seed)
    total_steps = cfg["epochs"] * math.ceil(len(texts) / batch_size)
    scheduler = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
        optimizer,
        num_warmup_steps=max(1, int(0.06 * total_steps)),
        num_training_steps=total_steps,
    )
    model.train()
    for _ in range(cfg["epochs"]):
        order = torch.randperm(len(texts), generator=generator).tolist()
        for start in range(0, len(texts), batch_size):
            batch = order[start : start + batch_size]
            encoded = tokenizer(
                [texts[i] for i in batch],
                truncation=True,
                max_length=max_seq_len,
                padding=True,
                return_tensors="pt",
            ).to(device)
            targets = torch.tensor([labels[i] for i in batch], device=device)
            optimizer.zero_grad()
            loss = model(**encoded, labels=targets).loss
            if not bool(torch.isfinite(loss)):
                msg = f"non-finite training loss for {model_id} on {device}"
                raise _NaNLossError(msg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
    return model


def finetune_transformer(
    model_id: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    cfg: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Fine-tune a transformer backbone on the canonical task; save model + tokenizer.

    ``cfg`` is the ``finetune`` block (``lr``, ``batch_size``, ``epochs``, ``max_seq_len``).
    Training runs in fp32 (see :func:`_train_transformer`), which is what keeps the
    fp16-checkpoint backbones (deberta-v3) stable rather than diverging to NaN. As a safety
    net against any other divergence, a non-finite loss is caught and the model is retrained
    once on CPU; the device actually used is recorded in the returned metadata.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = _train_transformer(model_id, tokenizer, train_df, cfg, device, seed)
        used_device = device
    except _NaNLossError:
        if device.type == "cpu":
            raise
        logger.warning(
            "%s diverged to a non-finite loss on %s; retraining on CPU", model_id, device
        )
        used_device = torch.device("cpu")
        model = _train_transformer(model_id, tokenizer, train_df, cfg, used_device, seed)
    SurrogateModelDataset(str(output_dir)).save(
        {"kind": "transformer", "model": model, "tokenizer": tokenizer}
    )
    accuracy = _transformer_accuracy(
        model, tokenizer, val_df, cfg["max_seq_len"], cfg["batch_size"], used_device
    )
    return {
        "val_accuracy": accuracy,
        "num_params": int(sum(p.numel() for p in model.parameters())),
        "num_labels": 2,
        "device": str(used_device),
    }


def _transformer_accuracy(
    model: Any,
    tokenizer: Any,
    df: pd.DataFrame,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """Top-1 accuracy of a transformer classifier over a DataFrame split."""
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    if not texts:
        return 0.0
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                max_length=max_seq_len,
                padding=True,
                return_tensors="pt",
            ).to(device)
            predictions = model(**encoded).logits.argmax(dim=-1).tolist()
            correct += sum(int(p == labels[start + i]) for i, p in enumerate(predictions))
    return correct / len(texts)


def train_bilstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    *,
    params: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Train the from-scratch BiLSTM classifier; save weights, vocab, and config."""
    torch.manual_seed(seed)
    texts = train_df["text"].tolist()
    labels = train_df["label"].tolist()
    vocab = build_vocab(texts, params["max_vocab"])
    max_len = params["max_seq_len"]
    encoded = [encode(text, vocab, max_len) for text in texts]
    model = BiLSTMClassifier(
        len(vocab),
        params["embed_dim"],
        params["hidden_dim"],
        params["num_layers"],
        params["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])
    generator = torch.Generator().manual_seed(seed)
    batch_size = params["batch_size"]
    model.train()
    for _ in range(params["epochs"]):
        order = torch.randperm(len(encoded), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            inputs = pad_batch([encoded[i] for i in batch]).to(device)
            targets = torch.tensor([labels[i] for i in batch], device=device)
            optimizer.zero_grad()
            loss = cross_entropy(model(inputs), targets)
            # torch leaves Tensor.backward unannotated, tripping strict no-untyped-call
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    accuracy = _bilstm_accuracy(model, vocab, val_df, max_len, batch_size, device)
    config = {
        "vocab": vocab,
        "embed_dim": params["embed_dim"],
        "hidden_dim": params["hidden_dim"],
        "num_layers": params["num_layers"],
        "dropout": params["dropout"],
        "max_seq_len": max_len,
    }
    bundle = {"kind": "bilstm", "model": model, "config": config}
    SurrogateModelDataset(str(output_dir)).save(bundle)
    return {
        "val_accuracy": accuracy,
        "num_params": sum(p.numel() for p in model.parameters()),
        "num_labels": 2,
    }


def _bilstm_accuracy(
    model: BiLSTMClassifier,
    vocab: dict[str, int],
    df: pd.DataFrame,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """Top-1 accuracy of the BiLSTM classifier over a DataFrame split."""
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    if not texts:
        return 0.0
    encoded = [encode(text, vocab, max_len) for text in texts]
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            inputs = pad_batch(encoded[start : start + batch_size]).to(device)
            predictions = model(inputs).argmax(dim=-1).tolist()
            correct += sum(int(p == labels[start + i]) for i, p in enumerate(predictions))
    return correct / len(texts)
