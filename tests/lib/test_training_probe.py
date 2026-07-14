"""Tests for deterministic training-only CKA probe sampling."""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from transfer_risk.lib.training_probe import sample_balanced_training_probe


def test_sample_balanced_training_probe_is_deterministic_and_preserves_metadata() -> None:
    train = pd.DataFrame(
        [
            {
                "row_id": f"train-positive-{index}",
                "text": f"positive {index}",
                "label": 1,
                "source": "source-positive",
                "split_marker": "train",
            }
            for index in range(5)
        ]
        + [
            {
                "row_id": f"train-negative-{index}",
                "text": f"negative {index}",
                "label": 0,
                "source": "source-negative",
                "split_marker": "train",
            }
            for index in range(5)
        ]
    )

    first = sample_balanced_training_probe(train, n_per_label=3, seed=17)
    second = sample_balanced_training_probe(train, n_per_label=3, seed=17)

    pdt.assert_frame_equal(first, second)
    assert len(first) == 6
    assert first["label"].value_counts().to_dict() == {1: 3, 0: 3}
    assert list(first.columns) == list(train.columns)
    assert set(first["source"]) == {"source-positive", "source-negative"}
    assert set(first["split_marker"]) == {"train"}


def test_sample_balanced_training_probe_rejects_insufficient_class_rows() -> None:
    train = pd.DataFrame(
        [
            {"text": "positive", "label": 1, "source": "source-a"},
            {"text": "negative 1", "label": 0, "source": "source-b"},
            {"text": "negative 2", "label": 0, "source": "source-b"},
        ]
    )

    with pytest.raises(ValueError, match=r"label 1=1, label 0=2"):
        sample_balanced_training_probe(train, n_per_label=2, seed=17)


@pytest.mark.parametrize("n_per_label", [0, -1])
def test_sample_balanced_training_probe_requires_positive_sample_size(n_per_label: int) -> None:
    """Sampling cannot produce a balanced probe with zero or negative class size."""
    train = pd.DataFrame(
        [
            {"text": "positive", "label": 1, "source": "source-a"},
            {"text": "negative", "label": 0, "source": "source-b"},
        ]
    )

    with pytest.raises(ValueError, match="n_per_label must be positive"):
        sample_balanced_training_probe(train, n_per_label=n_per_label, seed=17)


@pytest.mark.parametrize("missing_column", ["text", "label", "source"])
def test_sample_balanced_training_probe_requires_canonical_training_columns(
    missing_column: str,
) -> None:
    """Training-only sampling requires text, canonical label, and source provenance."""
    train = pd.DataFrame([{"text": "positive", "label": 1, "source": "source-a"}]).drop(
        columns=missing_column
    )

    with pytest.raises(ValueError, match="missing required columns"):
        sample_balanced_training_probe(train, n_per_label=1, seed=17)


def test_sample_balanced_training_probe_rejects_insufficient_negative_class_rows() -> None:
    """A probe requires enough label-zero rows as well as enough label-one rows."""
    train = pd.DataFrame(
        [
            {"text": "positive 1", "label": 1, "source": "source-a"},
            {"text": "positive 2", "label": 1, "source": "source-a"},
            {"text": "negative", "label": 0, "source": "source-b"},
        ]
    )

    with pytest.raises(ValueError, match=r"label 1=2, label 0=1"):
        sample_balanced_training_probe(train, n_per_label=2, seed=17)
