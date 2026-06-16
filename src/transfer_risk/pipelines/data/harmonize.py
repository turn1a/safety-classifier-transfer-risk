"""Pure helpers for building the canonical prompt-injection dataset.

No network or Kedro here: the data node loads the HuggingFace splits and hands plain
records to these functions, which normalise each source's schema to the canonical
``(text, label, source)`` shape, deduplicate, and split. Every source has its own
columns and label scheme (verified on HF), so each is mapped by a per-source adapter.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace, for dedup keys only (not training text)."""
    return _WHITESPACE.sub(" ", text).strip().lower()


def adapt_source(rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> pd.DataFrame:
    """Map one source's raw rows to canonical ``(text, label, source)`` records.

    ``spec`` carries ``id`` and ``text_col`` plus either ``label_const`` (a fixed label
    for single-class sources) or ``label_col`` (optionally with a ``label_map`` from raw
    values to ``{0, 1}``).

    Raises:
        ValueError: If the expected text or label column is absent (schema drift).
    """
    source = spec["id"]
    text_col = spec["text_col"]
    label_const = spec.get("label_const")
    label_col = spec.get("label_col")
    label_map = spec.get("label_map")
    row_list = list(rows)
    if row_list:
        sample = row_list[0]
        if text_col not in sample:
            msg = f"{source}: expected text column {text_col!r}, found {list(sample)}"
            raise ValueError(msg)
        if label_const is None and label_col not in sample:
            msg = f"{source}: expected label column {label_col!r}, found {list(sample)}"
            raise ValueError(msg)
    records: list[dict[str, Any]] = []
    for row in row_list:
        text = row.get(text_col)
        if not isinstance(text, str) or not text.strip():
            continue
        if label_const is not None:
            label = int(label_const)
        elif label_col is not None:
            raw = row[label_col]
            label = int(label_map[raw]) if label_map is not None else int(raw)
        else:  # unreachable: the up-front check requires label_const or label_col
            msg = f"{source}: no label_const or label_col configured"
            raise ValueError(msg)
        records.append({"text": text, "label": label, "source": source})
    return pd.DataFrame.from_records(records, columns=["text", "label", "source"])


def clean_and_dedup(frame: pd.DataFrame, min_chars: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop near-empty prompts and exact (normalised) duplicates; return an audit.

    The audit records raw/kept counts, duplicates removed, how many normalised texts
    appear in more than one source (cross-source leakage), the final class balance, and
    per-source counts.
    """
    n_raw = len(frame)
    long_enough = frame[frame["text"].str.len() >= min_chars].copy()
    long_enough["_key"] = long_enough["text"].map(normalize_text)
    per_key_sources = long_enough.groupby("_key")["source"].nunique()
    n_cross = int((per_key_sources > 1).sum())
    dup_mask = long_enough.duplicated("_key", keep="first")
    deduped = long_enough.loc[~dup_mask].drop(columns="_key").reset_index(drop=True)
    audit: dict[str, Any] = {
        "n_raw": n_raw,
        "n_after_min_chars": len(long_enough),
        "n_duplicates_removed": int(dup_mask.sum()),
        "n_cross_source_overlaps": n_cross,
        "n_final": len(deduped),
        "label_balance": {str(k): int(v) for k, v in deduped["label"].value_counts().items()},
        "per_source": {str(k): int(v) for k, v in deduped["source"].value_counts().items()},
    }
    return deduped, audit


def stratified_split(
    frame: pd.DataFrame,
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Deterministic, label-stratified train/val/test split."""
    train_frame, holdout = train_test_split(
        frame,
        train_size=ratios["train"],
        stratify=frame["label"],
        random_state=seed,
    )
    rel_test = ratios["test"] / (ratios["val"] + ratios["test"])
    val_frame, test_frame = train_test_split(
        holdout,
        test_size=rel_test,
        stratify=holdout["label"],
        random_state=seed,
    )
    return {
        "train": train_frame.reset_index(drop=True),
        "val": val_frame.reset_index(drop=True),
        "test": test_frame.reset_index(drop=True),
    }
