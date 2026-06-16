"""Nodes for the data pipeline (SPEC.md §3.1 step 1, §6)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from datasets import load_dataset

from transfer_risk.lib.seeds import derive_seeds
from transfer_risk.pipelines.data.harmonize import (
    adapt_source,
    clean_and_dedup,
    stratified_split,
)

logger = logging.getLogger(__name__)


def build_canonical_dataset(params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the configured sources, harmonise schemas, and deduplicate.

    Args:
        params: The ``data`` parameter block (``train_sources``, ``min_chars``, ...).

    Returns:
        The canonical ``(text, label, source)`` table, plus an audit dict (row counts,
        duplicates removed, cross-source overlaps, class balance).
    """
    frames: list[pd.DataFrame] = []
    for spec in params["train_sources"]:
        dataset = load_dataset(spec["id"])
        rows = [dict(row) for split in dataset.values() for row in split]
        frames.append(adapt_source(rows, spec))
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
