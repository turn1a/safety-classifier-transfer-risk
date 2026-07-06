"""Attacks pipeline assembly: one node per (surrogate, recipe, example-shard), plus reduces.

The pool, recipes, eval-set size, and shard size are read at build time (``_dynamic``) to fan
out the sweep. Each shard node writes ``adversarial_shard.{surrogate}__{recipe}__{start}`` and a
reduce node concatenates a cell's shards into ``adversarial.{surrogate}__{recipe}`` (the
partition the transfer stage reads). Nodes are tagged by surrogate, recipe, and ``attacks`` for
grouping; victims are wired by kind (ONNX for transformers, torch for the BiLSTM). Run with
``--runner ParallelRunner --only-missing-outputs`` to parallelise and resume.
"""

import inspect
from collections.abc import Callable
from functools import partial, update_wrapper
from typing import Any

from kedro.pipeline import Pipeline, node

from transfer_risk.lib.sweep import auto_shard_size, cell_key, shard_key, shard_spans
from transfer_risk.pipelines._dynamic import attack_params, surrogate_specs
from transfer_risk.pipelines.attacks.nodes import attack_shard, reduce_cell


def _bind_shard(
    name: str,
    kind: str,
    use_onnx: bool,  # noqa: FBT001 (build-time identity flag, not a runtime boolean-trap arg)
    recipe: str,
    start: int,
    stop: int,
    label: str,
) -> Callable[..., list[dict[str, Any]]]:
    """Bind one shard's build-time identity into a picklable, introspectable node callable.

    ``attack_shard`` is fanned out per ``(surrogate, recipe, shard)`` by binding its keyword-only
    identity with :func:`functools.partial` — picklable, so ``ParallelRunner`` can ship it to a
    spawned worker. A bare partial has no ``__name__`` and defeats ``typing.get_type_hints``, so
    Kedro warns per node on both counts; here :func:`functools.update_wrapper` copies
    ``attack_shard``'s name, docstring, annotations, and ``__wrapped__`` (for hint resolution), and
    the reduced signature is pinned so ``inspect.signature`` keeps reporting this node's four real
    inputs instead of unwrapping to the ten-arg original.

    Args:
        name: Surrogate name bound into the shard.
        kind: Surrogate kind (selects the victim wrapper).
        use_onnx: Serve the victim from its ONNX graph rather than its torch checkpoint.
        recipe: TextAttack recipe key.
        start: Shard start index into the eval set.
        stop: Shard stop index (exclusive).
        label: Readable node name (e.g. ``attack_<surrogate>__<recipe>__<start>``).

    Returns:
        The bound ``attack_shard`` callable, wrapped so Kedro can name it and read its type hints.
    """
    fn = partial(
        attack_shard,
        name=name,
        kind=kind,
        use_onnx=use_onnx,
        recipe=recipe,
        start=start,
        stop=stop,
    )
    # Capture the reduced signature (four real inputs; bound identity as defaults) before
    # update_wrapper sets __wrapped__ — which would otherwise make inspect.signature unwrap to
    # attack_shard's full ten-arg signature.
    signature = inspect.signature(fn)
    update_wrapper(fn, attack_shard)
    for attr, value in (
        ("__signature__", signature),
        ("__name__", label),
        ("__qualname__", label),
    ):
        setattr(fn, attr, value)
    return fn


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
        # The victim defaults to the torch checkpoint (``surrogate__``): the box's aarch64
        # onnxruntime fails every transformer's fused ONNX attention with a MatMul dimension
        # mismatch. A transformer opts back into its faster ONNX graph only with ``victim: onnx``
        # (on a platform where that is verified); the BiLSTM is always torch.
        use_onnx = kind != "bilstm" and spec.get("victim") == "onnx"
        victim = f"onnx__{name}" if use_onnx else f"surrogate__{name}"
        inputs = ["task_splits", victim, "params:attacks", "params:seed"]
        for recipe in recipes:
            cell = cell_key(name, recipe)
            if single:
                start, stop = spans[0]
                nodes.append(
                    node(
                        _bind_shard(name, kind, use_onnx, recipe, start, stop, f"attack_{cell}"),
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
                        _bind_shard(name, kind, use_onnx, recipe, start, stop, f"attack_{key}"),
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
