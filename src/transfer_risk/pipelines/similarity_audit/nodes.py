"""Nodes for the explicit training-only CKA sensitivity pipeline.

This pipeline is intentionally separate from the original similarity and target-audit pipelines.
It samples only the dedicated local training dataframe, recomputes CKA against saved local models,
and compares those values with saved aggregate evidence without creating new attack data.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from transfer_risk.lib.similarity_sensitivity import (
    build_training_probe_sensitivity_summary as summarize_training_probe_sensitivity,
)
from transfer_risk.lib.similarity_sensitivity import export_training_probe_similarity
from transfer_risk.lib.training_probe import sample_balanced_training_probe

logger = logging.getLogger(__name__)

_TRAINING_PROBE_SIZE = 1600


def build_training_probe_set(
    training_split: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Build the fixed balanced CKA probe from the dedicated training dataframe.

    Args:
        training_split: Persisted dataframe containing training rows only.
        params: Similarity-audit parameters with fixed ``n_probe``.
        seed: Root reproducibility seed.

    Returns:
        Deterministically shuffled training-only probe with 800 rows of each canonical label.

    Raises:
        ValueError: If ``n_probe`` is not the fixed sensitivity size.
    """
    n_probe = int(params["n_probe"])
    if n_probe != _TRAINING_PROBE_SIZE:
        msg = f"similarity_audit.n_probe must be {_TRAINING_PROBE_SIZE}, got {n_probe}"
        raise ValueError(msg)
    probe = sample_balanced_training_probe(
        training_split,
        n_per_label=n_probe // 2,
        seed=seed,
    )
    logger.info(
        "Built training-only CKA probe with %d rows (%d label 1, %d label 0)",
        len(probe),
        int((probe["label"] == 1).sum()),
        int((probe["label"] == 0).sum()),
    )
    return probe


def build_public_training_probe_similarity(training_similarity: pd.DataFrame) -> pd.DataFrame:
    """Select the public training-probe CKA and corrected-DBS table.

    Args:
        training_similarity: Internal per-surrogate training-probe similarity table.

    Returns:
        Public three-column training-probe similarity table.
    """
    public_table = export_training_probe_similarity(training_similarity)
    logger.info("Prepared %d public training-probe similarity rows", len(public_table))
    return public_table


def build_training_probe_sensitivity_summary(  # noqa: PLR0913
    training_probe: pd.DataFrame,
    training_similarity: pd.DataFrame,
    training_thresholds: dict[str, Any],
    training_selection: dict[str, Any],
    original_similarity: pd.DataFrame,
    original_selection: dict[str, Any],
    original_cka_matrices: dict[str, Any],
    target_audit_summary: dict[str, Any],
    similarity_params: dict[str, Any],
    risk_params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Build the public post-hoc sensitivity summary from saved aggregate evidence.

    Args:
        training_probe: Saved training-only probe.
        training_similarity: Recomputed training-probe CKA/DBS table.
        training_thresholds: Thresholds calibrated from training-probe mean CKA.
        training_selection: Training-probe M1/M2 membership.
        original_similarity: Saved original similarity table.
        original_selection: Saved original M1/M2 membership.
        original_cka_matrices: Saved original matrices used for corrected DBS only.
        target_audit_summary: Final stable target-outcome audit aggregate.
        similarity_params: Original CKA settings.
        risk_params: Risk-stage ablation settings.
        seed: Root reproducibility seed.

    Returns:
        JSON-compatible post-hoc sensitivity summary with no raw attack or prompt data.
    """
    summary = summarize_training_probe_sensitivity(
        training_probe=training_probe,
        training_similarity=training_similarity,
        training_thresholds=training_thresholds,
        training_selection=training_selection,
        original_similarity=original_similarity,
        original_selection=original_selection,
        original_cka_matrices=original_cka_matrices,
        target_audit_summary=target_audit_summary,
        similarity_params=similarity_params,
        risk_params=risk_params,
        seed=seed,
    )
    logger.info(
        "Prepared post-hoc training-probe CKA sensitivity summary from %d probe rows",
        summary["probe_metadata"]["n_rows"],
    )
    return summary
