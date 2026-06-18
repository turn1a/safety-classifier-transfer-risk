"""Attacks pipeline assembly: one node per (surrogate, recipe, example-shard), plus reduces.

The pool, recipes, eval-set size, and shard size are read at build time (``_dynamic``) to fan
out the sweep. Each shard node writes ``adversarial_shard.{surrogate}__{recipe}__{start}`` and a
reduce node concatenates a cell's shards into ``adversarial.{surrogate}__{recipe}`` (the
partition the transfer stage reads). Nodes are tagged by surrogate, recipe, and ``attacks`` for
grouping; victims are wired by kind (ONNX for transformers, torch for the BiLSTM). Run with
``--runner ParallelRunner --only-missing-outputs`` to parallelise and resume.
"""

from functools import partial

from kedro.pipeline import Pipeline, node

from transfer_risk.lib.sweep import cell_key, shard_key, shard_spans
from transfer_risk.pipelines._dynamic import attack_params, surrogate_specs
from transfer_risk.pipelines.attacks.nodes import attack_shard, reduce_cell


def create_pipeline() -> Pipeline:
    """Assemble the attacks pipeline (per-shard attack nodes + per-cell reduce nodes)."""
    specs = surrogate_specs()
    params = attack_params()
    recipes = list(params["recipes"])
    eval_size = int(params["eval_set_size"])
    shard_size = int(params.get("shard_size") or eval_size) or 1
    spans = shard_spans(eval_size, shard_size)
    nodes = []
    for spec in specs:
        name, kind = spec["name"], spec["kind"]
        victim = f"surrogate__{name}" if kind == "bilstm" else f"onnx__{name}"
        for recipe in recipes:
            shard_outputs = []
            for start, stop in spans:
                key = shard_key(name, recipe, start)
                nodes.append(
                    node(
                        partial(
                            attack_shard,
                            name=name,
                            kind=kind,
                            recipe=recipe,
                            start=start,
                            stop=stop,
                        ),
                        inputs=["task_splits", victim, "params:attacks", "params:seed"],
                        outputs=f"adversarial_shard__{key}",
                        name=f"attack_{key}",
                        tags=[name, recipe, "attacks"],
                    )
                )
                shard_outputs.append(f"adversarial_shard__{key}")
            cell = cell_key(name, recipe)
            nodes.append(
                node(
                    reduce_cell,
                    inputs=shard_outputs,
                    outputs=f"adversarial__{cell}",
                    name=f"reduce_{cell}",
                    tags=[name, recipe, "attacks"],
                )
            )
    return Pipeline(nodes)
