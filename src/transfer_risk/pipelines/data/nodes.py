"""Nodes for the data pipeline (SPEC.md §3.1 step 1, §6)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from transfer_risk.lib.seeds import derive_seeds
from transfer_risk.pipelines.data.harmonize import (
    adapt_source,
    clean_and_dedup,
    stratified_split,
)

logger = logging.getLogger(__name__)


def _flatten_rows(dataset: Any) -> list[dict[str, Any]]:
    """Flatten a HuggingFace ``DatasetDict`` (or bare ``Dataset``) into a list of row dicts."""
    splits = dataset.values() if hasattr(dataset, "values") else [dataset]
    return [dict(row) for split in splits for row in split]


def build_canonical_dataset(
    raw_deepset: Any,
    raw_jackhhao: Any,
    raw_lakera: Any,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Harmonise the catalog-loaded HuggingFace sources and deduplicate.

    The three raw sources are ``HFDataset`` catalog inputs, in the same order as
    ``params["train_sources"]``; each is adapted to ``(text, label)`` by its spec.

    Args:
        raw_deepset: deepset/prompt-injections (text + label).
        raw_jackhhao: jackhhao/jailbreak-classification (prompt + type).
        raw_lakera: Lakera/gandalf_ignore_instructions (text; all injections).
        params: The ``data`` parameter block (``train_sources``, ``min_chars``, ...).

    Returns:
        The canonical ``(text, label, source)`` table, plus an audit dict (row counts,
        duplicates removed, cross-source overlaps, class balance).
    """
    loaded = [raw_deepset, raw_jackhhao, raw_lakera]
    frames = [
        adapt_source(_flatten_rows(dataset), spec)
        for dataset, spec in zip(loaded, params["train_sources"], strict=True)
    ]
    combined = pd.concat(frames, ignore_index=True)
    canonical, audit = clean_and_dedup(combined, min_chars=params["min_chars"])
    logger.info("Canonical prompt-injection dataset built: %s", audit)
    return canonical, audit


def split_dataset(
    canonical: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Split the canonical dataset into deterministic, stratified train/val/test parts.

    Args:
        canonical: The canonical ``(text, label, source)`` table.
        params: The ``data`` parameter block (``splits``).
        seed: The project root seed; its NumPy child seed drives the split.

    Returns:
        Mapping with ``train`` / ``val`` / ``test`` DataFrames.
    """
    numpy_seed = derive_seeds(seed).numpy
    return stratified_split(canonical, params["splits"], numpy_seed)
