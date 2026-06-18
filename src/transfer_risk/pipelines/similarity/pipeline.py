"""Similarity pipeline assembly: per-surrogate CKA nodes generated from the configured pool.

The target's representations are computed once and reused; each surrogate gets its own
``cka_{name}`` node (tagged for grouping), and a reduce step assembles the matrices before the
DBS/threshold/selection steps. Tags (not namespaces) keep the ``cka__{name}`` factory names.
"""

from functools import partial

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines._dynamic import surrogate_specs
from transfer_risk.pipelines.similarity.nodes import (
    build_probe_set,
    calibrate_thresholds,
    compute_cka,
    compute_target_reps,
    reduce_cka,
    reduce_similarity,
    select_surrogates,
)


def create_pipeline() -> Pipeline:
    """Assemble the similarity pipeline (one CKA node per surrogate, then reduce + calibrate)."""
    specs = surrogate_specs()
    names = [spec["name"] for spec in specs]
    nodes = [
        node(
            build_probe_set,
            inputs=["canonical_dataset", "params:similarity", "params:seed"],
            outputs="probe_set",
            name="build_probe_set",
            tags=["similarity"],
        ),
        node(
            compute_target_reps,
            inputs=["target_model", "probe_set", "params:similarity", "params:device"],
            outputs="target_reps",
            name="compute_target_reps",
            tags=["similarity"],
        ),
    ]
    for spec in specs:
        name = spec["name"]
        nodes.append(
            node(
                partial(compute_cka, kind=spec["kind"]),
                inputs=[
                    f"surrogate__{name}",
                    "target_reps",
                    "probe_set",
                    "params:similarity",
                    "params:device",
                ],
                outputs=f"cka__{name}",
                name=f"cka_{name}",
                tags=[name, "similarity"],
            )
        )
    nodes.append(
        node(
            partial(reduce_cka, names=names),
            inputs=[f"cka__{name}" for name in names],
            outputs="cka_matrices",
            name="reduce_cka",
            tags=["similarity"],
        )
    )
    nodes += [
        node(
            reduce_similarity,
            inputs=["cka_matrices", "params:similarity"],
            outputs="similarity_table",
            name="reduce_similarity",
            tags=["similarity"],
        ),
        node(
            calibrate_thresholds,
            inputs=["similarity_table", "params:similarity"],
            outputs="thresholds",
            name="calibrate_thresholds",
            tags=["similarity"],
        ),
        node(
            select_surrogates,
            inputs=["similarity_table", "thresholds"],
            outputs="surrogate_selection",
            name="select_surrogates",
            tags=["similarity"],
        ),
    ]
    return Pipeline(nodes)
