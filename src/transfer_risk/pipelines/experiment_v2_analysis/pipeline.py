"""Assemble target-free held-out v2 analysis with a fresh fail-closed preflight."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.experiment_v2_analysis.nodes import analyse_v2_records
from transfer_risk.pipelines.experiment_v2_preflight.pipeline import create_pipeline as preflight


def create_pipeline() -> Pipeline:
    """Return aggregate-only rerun-record analysis with no training or inference nodes.

    Returns:
        The explicit prospective preflight and supplied-record analysis pipeline.
    """
    return preflight() + Pipeline(
        [
            node(
                analyse_v2_records,
                inputs=[
                    "v2_preflight_receipt",
                    "params:experiment_v2",
                    "v2_corpus_evidence",
                    "v2_registration_evidence",
                    "v2_calibration_records",
                    "v2_evaluation_records",
                    "v2_selection_lock",
                    "params:seed",
                ],
                outputs="v2_public_summary",
                name="v2_analyse_records",
            )
        ],
        tags="experiment_v2",
    )
