"""Tests for the explicit training-only CKA preparation pipeline."""

from __future__ import annotations

import pandas as pd

from transfer_risk.pipelines.similarity_audit_prepare.nodes import extract_training_split
from transfer_risk.pipelines.similarity_audit_prepare.pipeline import create_pipeline


def _split(prefix: str) -> pd.DataFrame:
    """Build a synthetic split without reading private data."""
    return pd.DataFrame(
        {
            "row_id": [f"{prefix}-0", f"{prefix}-1"],
            "text": [f"{prefix} prompt 0", f"{prefix} prompt 1"],
            "label": [0, 1],
            "split_marker": [prefix, prefix],
        }
    )


def test_extract_training_split_returns_only_the_training_frame() -> None:
    """The preparatory boundary discards non-training split frames."""
    training = _split("train")
    training_split = extract_training_split(
        {
            "train": training,
            "val": _split("val"),
            "test": _split("test"),
        }
    )

    assert training_split is not training
    pd.testing.assert_frame_equal(training_split, training)
    assert set(training_split["split_marker"]) == {"train"}
    assert not training_split["row_id"].str.startswith(("val-", "test-")).any()


def test_prepare_pipeline_reads_the_split_mapping_once_and_writes_training_split() -> None:
    """The only persisted preparation output is the training dataframe."""
    pipeline = create_pipeline()

    assert pipeline.inputs() == {"task_splits"}
    assert pipeline.outputs() == {"training_split"}
    assert {node.name for node in pipeline.nodes} == {"extract_training_split"}
    assert pipeline.nodes[0].inputs == ["task_splits"]
    assert pipeline.nodes[0].outputs == ["training_split"]
