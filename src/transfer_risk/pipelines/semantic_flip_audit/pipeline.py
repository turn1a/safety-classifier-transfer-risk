"""Explicit saved-data stages, excluded from measurement and inference pipelines."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.paper_reporting.nodes import build_checkpoint_comparison
from transfer_risk.pipelines.semantic_flip_audit.nodes import (
    finalize_flip_review,
    pending_flip_review,
    prepare_flip_cohort,
    render_flip_tools,
)


def create_tools_pipeline() -> Pipeline:
    """Render tools from the frozen cohort without resampling or reading annotations."""
    return Pipeline(
        [
            node(
                render_flip_tools,
                ["semantic_flip_cohort", "semantic_review_html_template"],
                ["semantic_flip_tool_reviewer_1", "semantic_flip_tool_reviewer_2"],
                name="render_semantic_flip_tools",
            )
        ]
    )


def create_pipeline() -> Pipeline:
    """Freeze a checked private cohort and publish only new pending aggregates."""
    return (
        Pipeline(
            [
                node(
                    build_checkpoint_comparison,
                    [
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
                    "semantic_flip_checked_context",
                    name="validate_semantic_flip_saved_context",
                ),
                node(
                    prepare_flip_cohort,
                    [
                        "semantic_audit_private_pairs",
                        "semantic_audit_sample_linkage",
                        "retrospective_predictions__protectai-v2",
                        "pub_retrospective_cells__protectai-v2",
                        "pub_target_audit_cells",
                        "pub_results_manifest",
                        "semantic_flip_checked_context",
                        "params:semantic_flip_audit",
                        "params:seed",
                    ],
                    "semantic_flip_cohort",
                    name="prepare_semantic_flip_cohort",
                ),
                node(
                    pending_flip_review,
                    ["semantic_flip_cohort", "pub_surrogate_summary"],
                    "pub_semantic_flip_preparation",
                    name="pending_semantic_flip_review",
                ),
            ]
        )
        + create_tools_pipeline()
    )


def create_review_pipeline() -> Pipeline:
    """Summarize explicitly imported labels separately from preparation outputs."""
    return Pipeline(
        [
            node(
                finalize_flip_review,
                ["semantic_flip_cohort", "semantic_flip_annotations", "pub_surrogate_summary"],
                ["semantic_flip_consensus", "pub_semantic_flip_review"],
                name="finalize_semantic_flip_review",
            )
        ]
    )
