"""Data pipeline assembly."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.data.nodes import build_canonical_dataset, split_dataset


def create_pipeline() -> Pipeline:
    """Assemble the data pipeline."""
    return Pipeline(
        [
            node(
                build_canonical_dataset,
                inputs="params:data",
                outputs=["canonical_dataset", "dataset_audit"],
                name="build_canonical_dataset",
            ),
            node(
                split_dataset,
                inputs=["canonical_dataset", "params:data", "params:seed"],
                outputs="task_splits",
                name="split_dataset",
            ),
        ]
    )
