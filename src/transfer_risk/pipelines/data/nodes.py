"""Nodes for the data pipeline (placeholders — see ``SPEC.md`` §3.1, §6)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd


def build_canonical_dataset(params: dict[str, Any]) -> pd.DataFrame:
    """Build the deduplicated canonical prompt-injection dataset from configured sources."""
    raise NotImplementedError


def split_dataset(canonical: pd.DataFrame, params: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Split the canonical dataset into deterministic train/val/test partitions."""
    raise NotImplementedError
