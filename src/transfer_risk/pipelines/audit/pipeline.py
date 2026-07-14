"""Audit pipeline assembly."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.audit.nodes import finalize_target_audit, run_target_audit


def create_pipeline() -> Pipeline:
    """Assemble target inference followed by deterministic audit finalization."""
    return Pipeline(
        [
            node(
                run_target_audit,
                inputs=[
                    "adversarial_examples",
                    "target_model",
                    "task_splits",
                    "master_results_table",
                    "params:transfer",
                    "params:audit",
                    "params:device",
                ],
                outputs=[
                    "target_audit_raw_cells",
                    "target_audit_raw_sources",
                    "target_audit_raw_context",
                ],
                name="run_target_audit",
                tags=["audit"],
            ),
            node(
                finalize_target_audit,
                inputs=[
                    "target_audit_raw_cells",
                    "target_audit_raw_sources",
                    "target_audit_raw_context",
                    "master_results_table",
                    "cka_matrices",
                    "params:similarity",
                    "surrogate_selection",
                    "params:risk",
                    "params:seed",
                ],
                outputs=["target_audit_cells", "target_audit_summary"],
                name="finalize_target_audit",
                tags=["audit", "audit_finalization"],
            ),
        ]
    )


def create_finalization_pipeline() -> Pipeline:
    """Assemble the saved-aggregate audit refresh without target-model inference."""
    return Pipeline(
        [
            node(
                finalize_target_audit,
                inputs=[
                    "target_audit_raw_cells",
                    "target_audit_raw_sources",
                    "target_audit_raw_context",
                    "master_results_table",
                    "cka_matrices",
                    "params:similarity",
                    "surrogate_selection",
                    "params:risk",
                    "params:seed",
                ],
                outputs=["target_audit_cells", "target_audit_summary"],
                name="finalize_target_audit",
                tags=["audit", "audit_finalization"],
            ),
        ]
    )
