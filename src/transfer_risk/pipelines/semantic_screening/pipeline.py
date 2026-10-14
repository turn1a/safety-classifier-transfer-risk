"""Wire catalog-owned review-sample models and private scores."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.semantic_screening.nodes import screen_review_sample


def create_pipeline() -> Pipeline:
    """Return an opt-in local stage with no training, attack search, or human-label inputs."""
    return Pipeline(
        [
            node(
                screen_review_sample,
                inputs=[
                    "semantic_audit_annotation_candidates",
                    "semantic_screen_encoder",
                    "semantic_screen_nli",
                    "params:semantic_screening",
                    "params:device",
                ],
                outputs=["semantic_screen_private_scores", "pub_semantic_screen_summary"],
                name="screen_review_sample",
                tags=["semantic_screening"],
            )
        ]
    )
