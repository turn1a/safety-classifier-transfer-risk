"""Assemble manuscript reporting from public aggregate datasets only."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.paper_reporting.nodes import (
    build_checkpoint_comparison,
    build_objective_sensitivity,
    build_paper_tables,
    build_pool_sensitivity,
    build_review_reporting,
    build_strengthening_tables,
    plot_audit_flow,
    plot_dbs_geometry,
    plot_objective_sensitivity,
    plot_probe_stability,
    plot_recipe_heatmap,
)


def create_pipeline() -> Pipeline:
    """Return a target-free pipeline with no attack, checkpoint, or text inputs."""
    return Pipeline(
        [
            node(
                build_review_reporting,
                inputs=["pub_semantic_flip_preparation", "pub_semantic_flip_review"],
                outputs=["paper_flip_review_table", "fig_flip_review_coverage"],
                name="paper_flip_review_reporting",
            ),
            node(
                build_objective_sensitivity,
                inputs=[
                    "pub_target_audit_cells",
                    "pub_surrogate_summary",
                    "pub_results_manifest",
                    "params:attacks",
                    "params:paper_reporting",
                ],
                outputs="pub_pool_objective_sensitivity",
                name="paper_objective_sensitivity",
            ),
            node(
                build_checkpoint_comparison,
                inputs=[
                    "pub_retrospective_cells__protectai-v2",
                    "pub_retrospective_cells__deepset-deberta-injection",
                    "pub_retrospective_cells__llama-prompt-guard-22m",
                    "pub_surrogate_summary",
                    "pub_retrospective_audit_summary",
                    "pub_results_manifest",
                    "params:attacks",
                    "pub_source_baseline_check",
                    "pub_detector_baseline_summary",
                ],
                outputs="pub_checkpoint_comparison",
                name="paper_checkpoint_comparison",
            ),
            node(
                build_strengthening_tables,
                inputs=[
                    "pub_pool_objective_sensitivity",
                    "pub_checkpoint_comparison",
                    "pub_detector_baseline_summary",
                    "pub_original_policy_coverage",
                ],
                outputs=[
                    "paper_objective_table",
                    "paper_baseline_table",
                    "paper_checkpoint_table",
                    "paper_coverage_table",
                ],
                name="paper_strengthening_tables",
            ),
            node(
                plot_objective_sensitivity,
                inputs=["pub_pool_objective_sensitivity", "pub_checkpoint_comparison"],
                outputs="fig_objective_sensitivity",
                name="paper_objective_figure",
            ),
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
