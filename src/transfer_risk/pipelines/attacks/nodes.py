"""Nodes for the attacks pipeline (SPEC.md §8).

The sweep is generated as one node per ``(surrogate, recipe, example-shard)`` plus one reduce
node per cell. Each shard node attacks a slice of the eval set with one recipe against one
victim and writes its partition through the catalog; ParallelRunner runs the shards across
cores and ``--only-missing-outputs`` resumes by skipping partitions already written. The victim
is wired at build time: by default every surrogate is served from its torch checkpoint
(``surrogate.{name}``), because the box's aarch64 onnxruntime fails transformer ONNX attention; a
transformer opts into its faster ONNX graph (``onnx.{name}``) with ``victim: onnx``. TextAttack runs
in-process via the ``turn1a/TextAttack`` fork; the runner binds its device at import, so the
CPU/thread/encoder env is set before the runner is imported inside the worker process.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


def attack_shard(  # noqa: PLR0913 (a generated per-shard node: catalog inputs + bound identity)
    splits: dict[str, pd.DataFrame],
    victim_path: str,
    attacks_params: dict[str, Any],
    seed: int,
    *,
    name: str,
    kind: str,
    use_onnx: bool,
    recipe: str,
    start: int,
    stop: int,
) -> list[dict[str, Any]]:
    """Attack the ``examples[start:stop]`` slice for one ``(surrogate, recipe)`` — one pool task.

    The eval set is the front-loaded injection prompts from the test split, truncated to
    ``max_prompt_chars`` and capped at ``eval_set_size``; the build-time shard span selects this
    node's slice. The per-shard seed is ``seed + start`` (the slice's absolute start), so a fixed
    ``shard_size`` and ``seed`` reproduce exactly. Loading the victim here (not in the parent)
    keeps the pickled task payload to plain data.

    Args:
        splits: train/val/test DataFrames; the eval set is drawn from ``test``.
        victim_path: local directory of the victim, materialised from the catalog — a torch
            checkpoint by default, or the ONNX graph for a ``victim: onnx`` transformer.
        attacks_params: the ``attacks`` block (eval_set_size, max_prompt_chars, max_seq_len,
            query_budget, semantic_encoder).
        seed: root seed; this shard attacks with ``seed + start``.
        name: the surrogate name (for logging/provenance), bound at build time.
        kind: the surrogate kind, bound at build time (selects the victim wrapper).
        use_onnx: serve the victim from its ONNX graph (True, ``victim: onnx`` transformers) or its
            torch checkpoint (False — the default, and always for the BiLSTM), bound at build time.
        recipe: the TextAttack recipe key, bound at build time.
        start: shard start index into the eval set, bound at build time.
        stop: shard stop index (exclusive), bound at build time.

    Returns:
        One record per attacked example in the shard (see :func:`runner.run_recipe`).
    """
    # Set these before importing the runner: the fork binds its device at import time, and the
    # worker process inherits this env. CPU + single-threaded so N pool workers use N cores.
    os.environ["TA_DEVICE"] = "cpu"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TA_SENTENCE_ENCODER"] = str(
        attacks_params.get("semantic_encoder", "sentence-transformers")
    )
    import torch  # noqa: PLC0415

    from transfer_risk.pipelines.attacks.runner import build_wrapper, run_recipe  # noqa: PLC0415

    torch.set_num_threads(1)
    eval_size = int(attacks_params["eval_set_size"])
    max_chars = int(attacks_params.get("max_prompt_chars") or 10**9)
    query_budget = int(attacks_params["query_budget"])
    max_seq_len = int(attacks_params.get("max_seq_len", 256))
    test_df = splits["test"]
    injections = test_df.loc[test_df["label"] == 1, "text"].head(eval_size).tolist()
    examples = [{"text": text[:max_chars], "label": 1} for text in injections]
    if use_onnx:
        wrapper = build_wrapper(
            {"kind": kind}, torch.device("cpu"), onnx_dir=victim_path, max_seq_len=max_seq_len
        )
    else:
        # The default: the BiLSTM (its own wrapper) or a transformer served from its torch
        # checkpoint (the box's aarch64 onnxruntime cannot run transformer ONNX attention).
        # build_wrapper returns a BiLSTMModelWrapper for kind ``bilstm``, else loads the checkpoint
        # and wraps it as a HuggingFaceModelWrapper.
        wrapper = build_wrapper({"kind": kind, "source": victim_path}, torch.device("cpu"))
    return run_recipe(
        wrapper, recipe, examples[start:stop], query_budget=query_budget, seed=seed + start
    )


def reduce_cell(*shard_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Concatenate a cell's shard records (in shard order) into its partition.

    The shard inputs are wired in ascending start order, so flattening preserves the eval-set
    order. The cell's ``(surrogate, recipe)`` identity is carried by the partition filename
    (``<surrogate>__<recipe>.pkl``), which the transfer stage parses, so the records stay minimal.
    """
    return [record for shard in shard_records for record in shard]
