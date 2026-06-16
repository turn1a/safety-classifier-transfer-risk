"""Models pipeline assembly."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.models.nodes import build_surrogate_registry, prepare_surrogates


def create_pipeline() -> Pipeline:
    """Assemble the models pipeline."""
    return Pipeline(
        [
            node(
                build_surrogate_registry,
                inputs="params:models",
                outputs="surrogate_registry",
                name="build_surrogate_registry",
            ),
            node(
                prepare_surrogates,
                inputs=["task_splits", "surrogate_registry", "params:models"],
                outputs="surrogate_checkpoints",
                name="prepare_surrogates",
            ),
        ]
    )
