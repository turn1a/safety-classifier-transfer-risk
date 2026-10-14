"""Assemble v2 selection freezing without held-out outcome inputs."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.experiment_v2_preflight.pipeline import create_pipeline as preflight
from transfer_risk.pipelines.experiment_v2_selection.nodes import freeze_v2_selectors


def create_pipeline() -> Pipeline:
    """Return preflight plus calibration-only source-selection freezing.

    Returns:
        A target-free selection pipeline with no evaluation evidence input.
    """
    return preflight() + Pipeline(
        [
            node(
                freeze_v2_selectors,
                inputs=[
                    "v2_preflight_receipt",
                    "params:experiment_v2",
                    "v2_corpus_evidence",
                    "v2_registration_evidence",
                    "v2_calibration_records",
                    "params:seed",
                ],
                outputs="v2_selection_lock",
                name="v2_freeze_selectors",
            )
        ],
        tags="experiment_v2",
    )
