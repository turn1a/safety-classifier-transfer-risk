"""Nodes for the transfer pipeline (placeholders — see ``SPEC.md`` §3.1 step 5)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd


def evaluate_transfer(adversarial_examples: dict[str, Any], params: dict[str, Any]) -> pd.DataFrame:
    """Evaluate adversarial examples against the frozen target; record transfer rates."""
    raise NotImplementedError


def assemble_results_table(
    transfer_results: pd.DataFrame,
    similarity_table: pd.DataFrame,
    registry: dict[str, Any],
) -> pd.DataFrame:
    """Join transfer rates with similarity and model features into the master results table."""
    raise NotImplementedError
