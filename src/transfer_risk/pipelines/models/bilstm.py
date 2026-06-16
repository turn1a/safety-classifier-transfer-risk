"""From-scratch BiLSTM-with-attention classifier — the non-transformer floor.

A word-level tokeniser, a 2-layer bidirectional LSTM, additive attention pooling, and
a binary head, trained on the canonical task (label 1 = injection). ``representations``
exposes the embedding- and sequence-level pooled vectors so the similarity stage can
compute CKA for this architecturally distinct model (SPEC.md §5, §7).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

_TOKEN = re.compile(r"\w+")
PAD_ID = 0
UNK_ID = 1


def tokenize(text: str) -> list[str]:
    """Lowercase word-level tokenisation."""
    return _TOKEN.findall(text.lower())


def build_vocab(texts: Sequence[str], max_vocab: int) -> dict[str, int]:
    """Build a word -> id vocabulary (ids 0/1 reserved for pad/unknown)."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for word, _ in counts.most_common(max(0, max_vocab - len(vocab))):
        vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    """Encode text to token ids, truncated to ``max_len`` (never empty)."""
    ids = [vocab.get(tok, UNK_ID) for tok in tokenize(text)[:max_len]]
    return ids or [UNK_ID]


def pad_batch(sequences: list[list[int]]) -> torch.Tensor:
    """Right-pad a list of id sequences into a ``(batch, max_len)`` long tensor."""
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [PAD_ID] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.long)


class BiLSTMClassifier(nn.Module):
    """2-layer BiLSTM with additive attention pooling and a binary head."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Build the embedding, BiLSTM, attention, and classifier layers."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Linear(2 * hidden_dim, 1)
        self.classifier = nn.Linear(2 * hidden_dim, 2)

    def _features(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (embedding output, LSTM output, attention-pooled context)."""
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * weights).sum(dim=1)
        return embedded, lstm_out, context

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Classify a batch of token-id sequences into 2 logits."""
        _, _, context = self._features(input_ids)
        logits: torch.Tensor = self.classifier(context)
        return logits

    def representations(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        """Per-"layer" pooled vectors for CKA: mean embedding and mean LSTM output."""
        embedded, lstm_out, _ = self._features(input_ids)
        emb_pooled: torch.Tensor = embedded.mean(dim=1)
        seq_pooled: torch.Tensor = lstm_out.mean(dim=1)
        return [emb_pooled, seq_pooled]
