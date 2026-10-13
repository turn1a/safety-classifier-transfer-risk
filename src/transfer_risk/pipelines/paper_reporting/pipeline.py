"""Assemble manuscript reporting from public aggregate datasets only."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.paper_reporting.nodes import (
    build_paper_tables,
    build_pool_sensitivity,
    plot_audit_flow,
    plot_dbs_geometry,
    plot_probe_stability,
    plot_recipe_heatmap,
)


def create_pipeline() -> Pipeline:
    """Return a target-free pipeline with no attack, checkpoint, or text inputs."""
    return Pipeline(
        [
            node(
                plot_dbs_geometry,
                inputs="params:similarity",
                outputs="fig_dbs_geometry",
                name="paper_dbs_geometry",
            ),
            node(
                build_pool_sensitivity,
                inputs=["pub_target_audit_summary", "params:paper_reporting"],
                outputs="pub_surrogate_excluded_sensitivity",
                name="paper_pool_sensitivity",
            ),
            node(
                build_paper_tables,
                inputs=[
                    "pub_target_audit_summary",
                    "pub_target_audit_cells",
                    "pub_training_probe_similarity_sensitivity",
                    "pub_surrogate_excluded_sensitivity",
                    "pub_results_manifest",
                ],
                outputs=[
                    "paper_transition_table",
                    "paper_surrogate_table",
                    "paper_recipe_table",
                    "paper_sensitivity_table",
                ],
                name="paper_aggregate_tables",
            ),
            node(
                plot_audit_flow,
                inputs=["pub_target_audit_summary", "pub_run_metrics"],
                outputs="fig_audit_flow",
                name="paper_audit_flow",
            ),
            node(
                plot_recipe_heatmap,
                inputs="pub_target_audit_cells",
                outputs="fig_recipe_heatmap",
                name="paper_recipe_heatmap",
            ),
            node(
                plot_probe_stability,
                inputs=["pub_target_audit_summary", "pub_training_probe_similarity"],
                outputs="fig_probe_stability",
                name="paper_probe_stability",
            ),
        ],
        tags="paper_reporting",
    )
