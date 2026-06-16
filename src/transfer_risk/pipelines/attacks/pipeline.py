"""Attacks pipeline assembly."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.attacks.nodes import run_attacks


def create_pipeline() -> Pipeline:
    """Assemble the attacks pipeline."""
    return Pipeline(
        [
            node(
                run_attacks,
                inputs=[
                    "task_splits",
                    "surrogate_checkpoints",
                    "params:attacks",
                    "params:seed",
                ],
                outputs="adversarial_examples",
                name="run_attacks",
            ),
        ]
    )
