"""Unit tests for the pure data-harmonisation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from transfer_risk.pipelines.data.harmonize import (
    adapt_source,
    clean_and_dedup,
    normalize_text,
    stratified_split,
)


def test_normalize_text_collapses_case_and_space() -> None:
    assert normalize_text("  Ignore   ALL\nprevious ") == "ignore all previous"


def test_adapt_source_label_map() -> None:
    rows = [{"prompt": "hi", "type": "benign"}, {"prompt": "do x", "type": "jailbreak"}]
    spec = {
        "id": "s",
        "text_col": "prompt",
        "label_col": "type",
        "label_map": {"benign": 0, "jailbreak": 1},
    }
    df = adapt_source(rows, spec)
    assert list(df["label"]) == [0, 1]
    assert list(df["text"]) == ["hi", "do x"]
    assert set(df["source"]) == {"s"}


def test_adapt_source_label_const() -> None:
    spec = {"id": "g", "text_col": "text", "label_const": 1}
    df = adapt_source([{"text": "ignore instructions"}], spec)
    assert list(df["label"]) == [1]


def test_adapt_source_missing_label_column_raises() -> None:
    with pytest.raises(ValueError, match="label column"):
        adapt_source([{"text": "x"}], {"id": "s", "text_col": "text", "label_col": "missing"})


def test_clean_and_dedup_removes_normalized_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "text": ["Ignore all", "ignore   all", "benign prompt", "x"],
            "label": [1, 1, 0, 1],
            "source": ["a", "b", "a", "a"],
        }
    )
    deduped, audit = clean_and_dedup(frame, min_chars=3)
    assert audit["n_duplicates_removed"] == 1
    assert audit["n_cross_source_overlaps"] == 1
    assert "x" not in set(deduped["text"])
    assert audit["n_final"] == len(deduped)


def test_stratified_split_is_deterministic_and_sized() -> None:
    frame = pd.DataFrame(
        {"text": [f"t{i}" for i in range(100)], "label": [0, 1] * 50, "source": ["a"] * 100}
    )
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    first = stratified_split(frame, ratios, 7)
    second = stratified_split(frame, ratios, 7)
    assert (len(first["train"]), len(first["val"]), len(first["test"])) == (80, 10, 10)
    assert list(first["train"]["text"]) == list(second["train"]["text"])
