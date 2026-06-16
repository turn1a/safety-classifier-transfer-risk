"""Nodes for the attacks pipeline (placeholders — see ``SPEC.md`` §8)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd


def run_attacks(
    splits: dict[str, pd.DataFrame],
    checkpoints: dict[str, Any],
    selection: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Run TextAttack recipes on selected surrogates over a fixed eval set; log JSONL."""
    raise NotImplementedError
