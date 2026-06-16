"""Unit tests for the pure model helpers (BiLSTM tokeniser + registry validation)."""

from __future__ import annotations

import pytest

from transfer_risk.pipelines.models.bilstm import (
    PAD_ID,
    UNK_ID,
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


def test_validate_specs_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        validate_surrogate_specs([{"name": "x", "kind": "bilstm"}, {"name": "x", "kind": "bilstm"}])


def test_validate_specs_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        validate_surrogate_specs([{"name": "x", "kind": "magic"}])


def test_validate_specs_requires_id_for_transformer_kinds() -> None:
    with pytest.raises(ValueError, match="requires an 'id'"):
        validate_surrogate_specs([{"name": "x", "kind": "finetune"}])


def test_requires_gated_auth() -> None:
    assert requires_gated_auth([{"name": "a", "kind": "pretrained", "gated": True}]) is True
    assert requires_gated_auth([{"name": "a", "kind": "bilstm"}]) is False
