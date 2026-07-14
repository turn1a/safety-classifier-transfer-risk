"""Assembly for the explicit training-only CKA preparation pipeline."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.similarity_audit_prepare.nodes import extract_training_split


def create_pipeline() -> Pipeline:
    """Assemble the split-deserialization boundary for training-only CKA."""
    return Pipeline(
        [
            node(
                extract_training_split,
                inputs="task_splits",
                outputs="training_split",
                name="extract_training_split",
                tags=["similarity_audit_prepare"],
            ),
        ]
    )
