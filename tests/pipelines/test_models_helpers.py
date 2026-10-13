"""Unit tests for the pure model helpers (BiLSTM tokeniser + registry validation)."""

from __future__ import annotations

import pytest
import torch

from transfer_risk.pipelines.models.bilstm import (
    PAD_ID,
    UNK_ID,
    BiLSTMClassifier,
    build_vocab,
    encode,
    pad_batch,
    tokenize,
)
from transfer_risk.pipelines.models.registry import (
    requires_gated_auth,
    validate_surrogate_specs,
)


def test_tokenize_lowercases_and_splits_words() -> None:
    assert tokenize("Ignore, ALL previous!") == ["ignore", "all", "previous"]


def test_build_vocab_reserves_pad_unk_and_caps_size() -> None:
    vocab = build_vocab(["a a a b b c", "a"], max_vocab=4)
    assert vocab["<pad>"] == PAD_ID
    assert vocab["<unk>"] == UNK_ID
    assert len(vocab) == 4
    assert "a" in vocab  # most frequent token is kept


def test_encode_maps_unknown_and_is_never_empty() -> None:
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID, "hello": 2}
    assert encode("hello world", vocab, max_len=5) == [2, UNK_ID]
    assert encode("   ", vocab, max_len=5) == [UNK_ID]


def test_pad_batch_right_pads_to_max_length() -> None:
    assert pad_batch([[2, 3], [4]]).tolist() == [[2, 3], [4, PAD_ID]]


def test_bilstm_predictions_and_representations_ignore_batch_padding() -> None:
    """A sequence has the same logits and representations in different batches."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = BiLSTMClassifier(16, 4, 3, 2, 0.0).eval()
    short = pad_batch([[2, 3]])
    mixed = pad_batch([[2, 3], [4, 5, 6, 7, 8, 9]])
    with torch.no_grad():
        torch.testing.assert_close(model(short)[0], model(mixed)[0])
        for alone, batched in zip(
            model.representations(short), model.representations(mixed), strict=True
        ):
            torch.testing.assert_close(alone[0], batched[0])


def test_bilstm_legacy_policy_preserves_saved_checkpoint_behavior() -> None:
    """Archived checkpoints retain their original unmasked recurrence and pooling."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = BiLSTMClassifier(16, 4, 3, 2, 0.0, padding_policy="legacy_unmasked").eval()
    inputs = pad_batch([[2, 3], [4, 5, 6, 7]])
    with torch.no_grad():
        embedded = model.embedding(inputs)
        sequence, _ = model.lstm(embedded)
        weights = torch.softmax(model.attention(sequence), dim=1)
        expected = model.classifier((sequence * weights).sum(dim=1))
        torch.testing.assert_close(model(inputs), expected)
        representations = model.representations(inputs)
        torch.testing.assert_close(representations[0], embedded.mean(dim=1))
        torch.testing.assert_close(representations[1], sequence.mean(dim=1))


def test_finetune_seeds_fresh_heads_without_changing_backbone() -> None:
    """Head initialization is repeatable even when Hub loading preceded seeding."""
    from copy import deepcopy  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification  # noqa: PLC0415

    from transfer_risk.pipelines.models.training import _train_transformer  # noqa: PLC0415

    with torch.random.fork_rng():
        torch.manual_seed(0)
        first = DebertaV2ForSequenceClassification(
            DebertaV2Config(
                vocab_size=16,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                num_labels=2,
            )
        )
        second = deepcopy(first)
        with torch.no_grad():
            for module in (second.classifier, second.pooler):
                for parameter in module.parameters():
                    parameter.uniform_(-1, 1)
        backbone = {name: value.clone() for name, value in first.deberta.state_dict().items()}
        config = {"epochs": 0, "batch_size": 1, "max_seq_len": 8, "lr": 2e-5}
        data = pd.DataFrame({"text": ["fixture"], "label": [1]})
        for model in (first, second):
            _train_transformer(model, None, data, config, torch.device("cpu"), 77)
        for name, value in first.deberta.state_dict().items():
            torch.testing.assert_close(value, backbone[name])
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name])


def test_validate_specs_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        validate_surrogate_specs([{"name": "x", "kind": "bilstm"}, {"name": "x", "kind": "bilstm"}])


def test_validate_specs_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        validate_surrogate_specs([{"name": "x", "kind": "magic"}])


def test_validate_specs_accepts_transformer_specs_without_id() -> None:
    # The repo id now lives in the catalog (hub.{name}), not the spec, so an id-less
    # transformer spec is valid.
    validate_surrogate_specs(
        [{"name": "x", "kind": "finetune"}, {"name": "y", "kind": "pretrained"}]
    )


def test_requires_gated_auth() -> None:
    assert requires_gated_auth([{"name": "a", "kind": "pretrained", "gated": True}]) is True
    assert requires_gated_auth([{"name": "a", "kind": "bilstm"}]) is False
