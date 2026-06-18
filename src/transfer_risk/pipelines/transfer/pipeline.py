"""Transfer pipeline assembly.

The assemble node gathers the per-cell ``adversarial__{cell}`` partitions (generated from the
configured pool and recipes) into the single table the transfer node feeds to the frozen
target. Reading the cells as explicit inputs connects the attack reduce nodes to the transfer
stage in the DAG (so the attacks always run first).
"""

from functools import partial

from kedro.pipeline import Pipeline, node

from transfer_risk.lib.sweep import cell_key
from transfer_risk.pipelines._dynamic import attack_params, surrogate_specs
from transfer_risk.pipelines.transfer.nodes import (
    assemble_adversarial,
    assemble_results_table,
    evaluate_transfer,
)


def create_pipeline() -> Pipeline:
    """Assemble the transfer pipeline (gather cells, evaluate transfer, join the results table)."""
    specs = surrogate_specs()
    recipes = list(attack_params()["recipes"])
    cellkeys = [cell_key(spec["name"], recipe) for spec in specs for recipe in recipes]
    return Pipeline(
        [
            node(
                partial(assemble_adversarial, cellkeys=cellkeys),
                inputs=[f"adversarial__{cell}" for cell in cellkeys],
                outputs="adversarial_examples",
                name="assemble_adversarial",
                tags=["transfer"],
            ),
            node(
                evaluate_transfer,
                inputs=[
                    "adversarial_examples",
                    "target_model",
                    "params:transfer",
                    "params:device",
                ],
                outputs=["transfer_results", "transferred_examples"],
                name="evaluate_transfer",
                tags=["transfer"],
            ),
            node(
                assemble_results_table,
                inputs=["transfer_results", "similarity_table", "surrogate_registry"],
                outputs="master_results_table",
                name="assemble_results_table",
                tags=["transfer"],
            ),
        ]
    )
