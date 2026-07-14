"""Target-free public reporting pipeline for finalized target-outcome audit aggregates."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.audit_reporting.nodes import (
    build_public_target_audit_cells,
    build_public_target_audit_sources,
    build_public_target_audit_summary,
    plot_true_flip_ablation,
    plot_true_flip_scatter,
)
from transfer_risk.pipelines.reporting.nodes import (
    build_qualitative_examples,
    plot_transfer_scatter,
)


def create_pipeline() -> Pipeline:
    """Assemble explicit public audit reporting without any target-model dependency."""
    return Pipeline(
        [
            node(
                build_qualitative_examples,
                inputs=None,
                outputs="pub_qualitative_examples",
                name="build_redacted_qualitative_audit",
                tags=["audit_reporting"],
            ),
            node(
                plot_transfer_scatter,
                inputs="pub_master_results_table",
                outputs="fig_transfer_scatter",
                name="plot_historical_conditional_scatter",
                tags=["audit_reporting"],
            ),
            node(
                build_public_target_audit_cells,
                inputs="target_audit_cells",
                outputs="pub_target_audit_cells",
                name="build_public_target_audit_cells",
                tags=["audit_reporting"],
            ),
            node(
                build_public_target_audit_sources,
                inputs="target_audit_raw_sources",
                outputs="pub_target_audit_sources",
                name="build_public_target_audit_sources",
                tags=["audit_reporting"],
            ),
            node(
                build_public_target_audit_summary,
                inputs="target_audit_summary",
                outputs="pub_target_audit_summary",
                name="build_public_target_audit_summary",
                tags=["audit_reporting"],
            ),
            node(
                plot_true_flip_scatter,
                inputs=["target_audit_cells", "target_audit_summary"],
                outputs="fig_true_flip_scatter",
                name="plot_true_flip_scatter",
                tags=["audit_reporting"],
            ),
            node(
                plot_true_flip_ablation,
                inputs=["target_audit_summary", "surrogate_selection"],
                outputs="fig_true_flip_ablation",
                name="plot_true_flip_ablation",
                tags=["audit_reporting"],
            ),
        ]
    )
