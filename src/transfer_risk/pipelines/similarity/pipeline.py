"""Similarity pipeline assembly."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.similarity.nodes import (
    build_probe_set,
    calibrate_thresholds,
    compute_cka_matrices,
    reduce_similarity,
    select_surrogates,
)


def create_pipeline() -> Pipeline:
    """Assemble the similarity pipeline."""
    return Pipeline(
        [
            node(
                build_probe_set,
                inputs=["canonical_dataset", "params:similarity", "params:seed"],
                outputs="probe_set",
                name="build_probe_set",
            ),
            node(
                compute_cka_matrices,
                inputs=[
                    "probe_set",
                    "surrogate_checkpoints",
                    "surrogate_registry",
                    "params:similarity",
                    "params:device",
                ],
                outputs="cka_matrices",
                name="compute_cka_matrices",
            ),
            node(
                reduce_similarity,
                inputs=["cka_matrices", "params:similarity"],
                outputs="similarity_table",
                name="reduce_similarity",
            ),
            node(
                calibrate_thresholds,
                inputs=["similarity_table", "params:similarity"],
                outputs="thresholds",
                name="calibrate_thresholds",
            ),
            node(
                select_surrogates,
                inputs=["similarity_table", "thresholds"],
                outputs="surrogate_selection",
                name="select_surrogates",
            ),
        ]
    )
