"""Assembly for private preparation and separate human-label semantic review."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.semantic_audit.nodes import (
    pending_semantic_review,
    prepare_semantic_audit,
    render_semantic_review_tools,
    review_semantic_audit,
    summarize_semantic_valid_flip_yield,
)


def create_pipeline() -> Pipeline:
    """Assemble saved-attack preparation without reading any submitted annotations."""
    return Pipeline(
        [
            node(
                prepare_semantic_audit,
                inputs=[
                    "adversarial_examples",
                    "task_splits",
                    "params:semantic_audit",
                    "params:seed",
                ],
                outputs=[
                    "semantic_audit_private_pairs",
                    "semantic_audit_annotation_candidates",
                    "semantic_audit_sample_linkage",
                    "pub_semantic_fidelity_summary",
                ],
                name="prepare_semantic_audit",
                tags=["semantic_audit"],
            ),
            node(
                pending_semantic_review,
                inputs=["semantic_audit_private_pairs", "semantic_audit_sample_linkage"],
                outputs=[
                    "semantic_audit_pending_consensus",
                    "pub_semantic_review_preparation_summary",
                ],
                name="pending_semantic_review",
                tags=["semantic_audit"],
            ),
        ]
    )


def create_review_pipeline() -> Pipeline:
    """Assemble explicit human-label finalization with outputs separate from preparation."""
    return Pipeline(
        [
            node(
                review_semantic_audit,
                inputs=[
                    "semantic_audit_private_pairs",
                    "semantic_audit_sample_linkage",
                    "semantic_audit_annotations",
                    "params:semantic_audit",
                ],
                outputs=["semantic_audit_review_consensus", "pub_semantic_review_summary"],
                name="review_semantic_audit",
                tags=["semantic_audit_review"],
            ),
            node(
                summarize_semantic_valid_flip_yield,
                inputs=[
                    "semantic_audit_review_consensus",
                    "retrospective_predictions__protectai-v2",
                    "semantic_audit_sample_linkage",
                    "params:retrospective_audit.targets.protectai-v2",
                    "pub_target_audit_cells",
                    "pub_results_manifest",
                ],
                outputs="pub_semantic_valid_flip_yield_summary",
                name="summarize_semantic_valid_flip_yield",
                tags=["semantic_audit_review"],
            ),
        ]
    )


def create_tools_pipeline() -> Pipeline:
    """Render private human-review tools without regenerating the saved random sample."""
    return Pipeline(
        [
            node(
                render_semantic_review_tools,
                inputs=[
                    "semantic_audit_annotation_candidates",
                    "semantic_review_html_template",
                    "params:semantic_audit",
                ],
                outputs=[
                    "semantic_audit_review_tool_reviewer_1",
                    "semantic_audit_review_tool_reviewer_2",
                ],
                name="render_semantic_review_tools",
                tags=["semantic_review_tools"],
            ),
        ]
    )
