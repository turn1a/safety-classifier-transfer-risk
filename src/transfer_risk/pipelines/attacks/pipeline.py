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

from transfer_risk.lib.sweep import auto_shard_size, cell_key, shard_key, shard_spans
from transfer_risk.pipelines._dynamic import attack_params, surrogate_specs
from transfer_risk.pipelines.attacks.nodes import attack_shard, reduce_cell


def create_pipeline() -> Pipeline:
    """Assemble the attacks pipeline (per-shard attack nodes + per-cell reduce nodes).

    The shard count scales to the run box: with no explicit ``shard_size``, each cell is split
    into ``~shard_multiple * cores / n_cells`` shards (see :func:`auto_shard_size`), so a slow cell
    still spreads across cores without the node-count blow-up of a fixed tiny shard size. A cell
    that resolves to a single shard becomes one ``attack_{surrogate}__{recipe}`` node writing its
    cell partition directly, with no reduce.
    """
    specs = surrogate_specs()
    params = attack_params()
    recipes = list(params["recipes"])
    eval_size = int(params["eval_set_size"])
    explicit = params.get("shard_size")
    if explicit:
        shard_size = int(explicit)
    else:
        shard_size = auto_shard_size(
            eval_size,
            len(specs) * len(recipes),
            int(params["cores"]),
            int(params.get("shard_multiple", 2)),
        )
    spans = shard_spans(eval_size, shard_size)
    single = len(spans) == 1
    nodes = []
    for spec in specs:
        name, kind = spec["name"], spec["kind"]
        victim = f"surrogate__{name}" if kind == "bilstm" else f"onnx__{name}"
        inputs = ["task_splits", victim, "params:attacks", "params:seed"]
        for recipe in recipes:
            cell = cell_key(name, recipe)
            if single:
                start, stop = spans[0]
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
                        inputs=inputs,
                        outputs=f"adversarial__{cell}",
                        name=f"attack_{cell}",
                        tags=[name, recipe, "attacks"],
                    )
                )
                continue
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
                        inputs=inputs,
                        outputs=f"adversarial_shard__{key}",
                        name=f"attack_{key}",
                        tags=[name, recipe, "attacks"],
                    )
                )
                shard_outputs.append(f"adversarial_shard__{key}")
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
