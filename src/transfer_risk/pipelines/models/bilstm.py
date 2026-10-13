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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        *,
        padding_policy: str = "masked",
    ) -> None:
        """Build the classifier with masked padding or archived unmasked behavior.

        Args:
            vocab_size: Vocabulary size, including padding and unknown tokens.
            embed_dim: Embedding width.
            hidden_dim: Hidden width for each LSTM direction.
            num_layers: Number of bidirectional recurrent layers.
            dropout: Dropout between recurrent layers.
            padding_policy: ``masked`` for new models or ``legacy_unmasked`` for
                historical checkpoints whose padding affected model outputs.

        Raises:
            ValueError: If the padding policy is unknown.
        """
        super().__init__()
        if padding_policy not in {"masked", "legacy_unmasked"}:
            msg = f"unknown BiLSTM padding policy: {padding_policy!r}"
            raise ValueError(msg)
        self.padding_policy = padding_policy
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
        mask = input_ids != PAD_ID
        if self.padding_policy == "legacy_unmasked":
            lstm_out, _ = self.lstm(embedded)
            attention_logits = self.attention(lstm_out)
        else:
            lengths = mask.sum(dim=1)
            if bool((lengths == 0).any()):
                msg = "each BiLSTM input must contain at least one non-padding token"
                raise ValueError(msg)
            packed = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=input_ids.shape[1]
            )
            attention_logits = self.attention(lstm_out).masked_fill(
                ~mask.unsqueeze(-1), torch.finfo(lstm_out.dtype).min
            )
        weights = torch.softmax(attention_logits, dim=1)
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
        if self.padding_policy == "masked":
            mask = (input_ids != PAD_ID).unsqueeze(-1)
            lengths = mask.sum(dim=1)
            return [
                (embedded * mask).sum(dim=1) / lengths,
                (lstm_out * mask).sum(dim=1) / lengths,
            ]
        emb_pooled: torch.Tensor = embedded.mean(dim=1)
        seq_pooled: torch.Tensor = lstm_out.mean(dim=1)
        return [emb_pooled, seq_pooled]
