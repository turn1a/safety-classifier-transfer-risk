"""Models pipeline assembly: one node per surrogate, generated from the configured pool.

The surrogate specs are read at build time from config (``_dynamic.surrogate_specs``), so adding
a surrogate needs no pipeline edit. Each surrogate gets a train/materialise node writing its
``surrogate__{name}`` checkpoint and a ``surrogate_meta__{name}`` provenance fragment; each
transformer also gets an ONNX export node. Nodes are tagged with the surrogate name (and
``models``) for grouping in kedro-viz and selective runs — tags rather than namespaces, so the
factory dataset names (``surrogate__{name}`` etc.) are not rewritten.
"""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines._dynamic import surrogate_specs
from transfer_risk.pipelines.models.nodes import (
    build_surrogate_registry,
    export_onnx,
    materialize_surrogate,
    train_bilstm_surrogate,
    train_surrogate,
)


def create_pipeline() -> Pipeline:
    """Assemble the models pipeline (per-surrogate train/materialise + ONNX export nodes)."""
    nodes = [
        node(
            build_surrogate_registry,
            inputs="params:models",
            outputs="surrogate_registry",
            name="build_surrogate_registry",
            tags=["models"],
        )
    ]
    for spec in surrogate_specs():
        name, kind = spec["name"], spec["kind"]
        outputs = [f"surrogate__{name}", f"surrogate_meta__{name}"]
        if kind == "pretrained":
            nodes.append(
                node(
                    materialize_surrogate,
                    inputs=f"hub__{name}",
                    outputs=outputs,
                    name=f"materialize_{name}",
                    tags=[name, "models"],
                )
            )
        elif kind == "finetune":
            nodes.append(
                node(
                    train_surrogate,
                    inputs=[
                        f"hub__{name}",
                        "task_splits",
                        "params:models",
                        "params:device",
                        "params:seed",
                    ],
                    outputs=outputs,
                    name=f"train_{name}",
                    tags=[name, "models"],
                )
            )
        else:  # bilstm
            nodes.append(
                node(
                    train_bilstm_surrogate,
                    inputs=["task_splits", "params:models", "params:device", "params:seed"],
                    outputs=outputs,
                    name=f"train_{name}",
                    tags=[name, "models"],
                )
            )
        if kind != "bilstm":
            nodes.append(
                node(
                    export_onnx,
                    inputs=[f"surrogate__{name}", "params:models"],
                    outputs=f"onnx__{name}",
                    name=f"export_onnx_{name}",
                    tags=[name, "models"],
                )
            )
    return Pipeline(nodes)
