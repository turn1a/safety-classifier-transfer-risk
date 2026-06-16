"""Transfer pipeline assembly."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.transfer.nodes import assemble_results_table, evaluate_transfer


def create_pipeline() -> Pipeline:
    """Assemble the transfer pipeline."""
    return Pipeline(
        [
            node(
                evaluate_transfer,
                inputs=["adversarial_examples", "params:transfer"],
                outputs="transfer_results",
                name="evaluate_transfer",
            ),
            node(
                assemble_results_table,
                inputs=["transfer_results", "similarity_table", "surrogate_registry"],
                outputs="master_results_table",
                name="assemble_results_table",
            ),
        ]
    )
