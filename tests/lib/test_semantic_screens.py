import json

import numpy as np
import pandas as pd
import pytest

from transfer_risk.lib.semantic_screens import (
    cosine_pairs,
    entailment_probabilities,
    summarize_semantic_screens,
)


def test_embedding_cosine_normalizes_each_vector() -> None:
    assert cosine_pairs([[2, 0], [0, 3]], [[5, 0], [1, 0]]).tolist() == [1.0, 0.0]


@pytest.mark.parametrize(
    ("left", "right"),
    [([[0, 0]], [[1, 0]]), ([[float("nan"), 1]], [[1, 0]]), ([[1, 0]], [[1, 0, 0]])],
)
def test_embedding_cosine_rejects_invalid_vectors(left: list, right: list) -> None:
    with pytest.raises(ValueError):
        cosine_pairs(left, right)


def test_nli_uses_declared_class_order_and_stable_softmax() -> None:
    result = entailment_probabilities(
        [[1000, 1000, 1000]], {0: "neutral", 1: "contradiction", 2: "entailment"}
    )
    assert set(result) == {"neutral", "contradiction", "entailment"}
    assert result["entailment"][0] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "mapping", [{0: "benign", 1: "entailment", 2: "neutral"}, {0: "neutral", 1: "neutral"}]
)
def test_nli_rejects_unknown_or_incomplete_label_maps(mapping: dict) -> None:
    with pytest.raises(ValueError):
        entailment_probabilities([[1, 2, 3]], mapping)


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": ["private-a", "private-b"],
            "original_id": ["private-original", "private-original"],
            "embedding_cosine": [0.7, 0.9],
            "nli_min_entailment": [0.1, 0.4],
            "nli_max_contradiction": [0.2, 0.8],
            "embedding_truncated": [True, False],
            "nli_truncated": [False, False],
        }
    )


def test_screen_summary_exposes_coverage_without_private_rows_or_validity_labels() -> None:
    summary = summarize_semantic_screens(_scores(), {"encoder": "pinned-test"})
    assert summary["sample_pairs"] == 2
    assert summary["unique_originals"] == 1
    assert summary["coverage"]["embedding_truncated_pairs"] == 1
    assert summary["scores"]["embedding_cosine"]["median"] == pytest.approx(0.8)
    assert summary["semantic_validity_assessed"] is False
    serialized = json.dumps(summary, allow_nan=False)
    assert "private-a" not in serialized
    assert "private-original" not in serialized


@pytest.mark.parametrize("change", ["duplicate", "nonfinite", "out_of_range", "no_rows"])
def test_screen_summary_rejects_incomplete_or_invalid_evidence(change: str) -> None:
    scores = _scores()
    if change == "duplicate":
        scores.loc[1, "pair_id"] = scores.loc[0, "pair_id"]
    elif change == "nonfinite":
        scores.loc[0, "embedding_cosine"] = np.inf
    elif change == "out_of_range":
        scores.loc[0, "nli_min_entailment"] = 1.1
    else:
        scores = scores.iloc[:0]
    with pytest.raises(ValueError):
        summarize_semantic_screens(scores, {})
