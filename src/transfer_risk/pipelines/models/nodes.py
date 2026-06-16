"""Nodes for the models pipeline (placeholders — see ``SPEC.md`` §5, §7)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd


def build_surrogate_registry(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve the model-agnostic surrogate registry (HF ids / local checkpoint paths)."""
    raise NotImplementedError


def prepare_surrogates(
    splits: dict[str, pd.DataFrame],
    registry: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Collect pre-fine-tuned surrogates, fine-tune the rest, and train the BiLSTM outlier."""
    raise NotImplementedError
