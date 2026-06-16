"""Training procedures for the surrogate pool: transformer fine-tuning + the BiLSTM.

Both use a small explicit torch loop rather than the HF ``Trainer`` — ``model(**enc,
labels=...).loss`` is stable across transformers versions, and the loop keeps seeding
and the MPS device explicit (SPEC.md §7). Each function trains, saves a checkpoint, and
returns manifest metadata (validation accuracy, parameter count).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn.functional import cross_entropy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transfer_risk.datasets.surrogate_model_dataset import SurrogateModelDataset
from transfer_risk.pipelines.models.bilstm import BiLSTMClassifier, build_vocab, encode, pad_batch

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


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

    ``cfg`` is the ``finetune`` parameter block (``lr``, ``batch_size``, ``epochs``,
    ``max_seq_len``).
    """
    torch.manual_seed(seed)
    batch_size, max_seq_len = cfg["batch_size"], cfg["max_seq_len"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    texts = train_df["text"].tolist()
    labels = train_df["label"].tolist()
    generator = torch.Generator().manual_seed(seed)
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
            outputs = model(**encoded, labels=targets)
            outputs.loss.backward()
            optimizer.step()
    SurrogateModelDataset(str(output_dir)).save(
        {"kind": "transformer", "model": model, "tokenizer": tokenizer}
    )
    accuracy = _transformer_accuracy(model, tokenizer, val_df, max_seq_len, batch_size, device)
    return {
        "val_accuracy": accuracy,
        "num_params": int(sum(p.numel() for p in model.parameters())),
        "num_labels": 2,
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
