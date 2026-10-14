"""Assemble an explicit target-free v2 report-then-gate pipeline."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.experiment_v2_preflight.nodes import (
    gate_v2_contract,
    inspect_v2_contract,
)


def create_pipeline() -> Pipeline:
    """Return prospective validation outside every historical pipeline composite.

    Returns:
        Nodes that save blocker evidence before rejecting an incomplete protocol.
    """
    return Pipeline(
        [
            node(
                inspect_v2_contract,
                inputs=[
                    "params:experiment_v2",
                    "v2_corpus_evidence",
                    "v2_registration_evidence",
                    "params:seed",
                ],
                outputs=["v2_preflight_report", "v2_execution_manifest"],
                name="v2_inspect_contract",
            ),
            node(
                gate_v2_contract,
                inputs="v2_preflight_report",
                outputs="v2_preflight_receipt",
                name="v2_gate_contract",
            ),
        ],
        tags="experiment_v2",
    )
