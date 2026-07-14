"""Assembly for the explicit training-only CKA sensitivity pipeline.

The pipeline reads the dedicated training input, saved models, original CKA artifacts, and final
target-audit aggregates. It neither consumes attack records nor participates in normal stages.
"""

from __future__ import annotations

from functools import partial

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines._dynamic import surrogate_specs
from transfer_risk.pipelines.similarity.nodes import (
    calibrate_thresholds,
    compute_cka,
    compute_target_reps,
    reduce_cka,
    reduce_similarity,
    select_surrogates,
)
from transfer_risk.pipelines.similarity_audit.nodes import (
    build_public_training_probe_similarity,
    build_training_probe_sensitivity_summary,
    build_training_probe_set,
)


def create_pipeline() -> Pipeline:
    """Assemble a standalone CKA sensitivity run over the saved training split only."""
    specs = surrogate_specs()
    names = [spec["name"] for spec in specs]
    nodes = [
        node(
            build_training_probe_set,
            inputs=["training_split", "params:similarity_audit", "params:seed"],
            outputs="training_probe_set",
            name="build_training_probe_set",
            tags=["similarity_audit"],
        ),
        node(
            compute_target_reps,
            inputs=["target_model", "training_probe_set", "params:similarity", "params:device"],
            outputs="training_target_reps",
            name="compute_training_target_reps",
            tags=["similarity_audit"],
        ),
    ]
    for spec in specs:
        name = spec["name"]
        nodes.append(
            node(
                partial(compute_cka, kind=spec["kind"]),
                inputs=[
                    f"surrogate__{name}",
                    "training_target_reps",
                    "training_probe_set",
                    "params:similarity",
                    "params:device",
                ],
                outputs=f"training_cka__{name}",
                name=f"training_cka_{name}",
                tags=[name, "similarity_audit"],
            )
        )
    nodes.extend(
        [
            node(
                partial(reduce_cka, names=names),
                inputs=[f"training_cka__{name}" for name in names],
                outputs="training_cka_matrices",
                name="reduce_training_cka",
                tags=["similarity_audit"],
            ),
            node(
                reduce_similarity,
                inputs=["training_cka_matrices", "params:similarity"],
                outputs="training_similarity_table",
                name="reduce_training_similarity",
                tags=["similarity_audit"],
            ),
            node(
                calibrate_thresholds,
                inputs=["training_similarity_table", "params:similarity"],
                outputs="training_similarity_thresholds",
                name="calibrate_training_thresholds",
                tags=["similarity_audit"],
            ),
            node(
                select_surrogates,
                inputs=["training_similarity_table", "training_similarity_thresholds"],
                outputs="training_surrogate_selection",
                name="select_training_surrogates",
                tags=["similarity_audit"],
            ),
            node(
                build_public_training_probe_similarity,
                inputs="training_similarity_table",
                outputs="pub_training_probe_similarity",
                name="build_public_training_probe_similarity",
                tags=["similarity_audit"],
            ),
            node(
                build_training_probe_sensitivity_summary,
                inputs=[
                    "training_probe_set",
                    "training_similarity_table",
                    "training_similarity_thresholds",
                    "training_surrogate_selection",
                    "similarity_table",
                    "surrogate_selection",
                    "cka_matrices",
                    "target_audit_summary",
                    "params:similarity",
                    "params:risk",
                    "params:seed",
                ],
                outputs="pub_training_probe_similarity_sensitivity",
                name="build_training_probe_sensitivity_summary",
                tags=["similarity_audit"],
            ),
        ]
    )
    return Pipeline(nodes)
