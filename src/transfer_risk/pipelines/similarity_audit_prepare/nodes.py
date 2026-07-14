"""Nodes that materialize the training-only CKA input artifact."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd

logger = logging.getLogger(__name__)


def extract_training_split(task_splits: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Copy the saved training dataframe into the dedicated CKA input artifact.

    Args:
        task_splits: Persisted mapping of task-split dataframes.

    Returns:
        Independent dataframe containing only the training rows.

    Raises:
        ValueError: If the saved mapping has no training dataframe.
    """
    try:
        training_split = task_splits["train"]
    except KeyError as error:
        msg = "task_splits must contain a saved train split"
        raise ValueError(msg) from error

    extracted_split = training_split.copy()
    logger.info("Prepared dedicated training CKA input with %d rows", len(extracted_split))
    return extracted_split
