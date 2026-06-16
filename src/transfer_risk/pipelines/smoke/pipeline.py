"""Smoke pipeline assembly."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.smoke.nodes import record_environment


def create_pipeline() -> Pipeline:
    """Assemble the smoke wiring-check pipeline."""
    return Pipeline(
        [
            node(
                record_environment,
                inputs=["params:project", "params:seed"],
                outputs="smoke_report",
                name="record_environment",
            ),
        ]
    )
