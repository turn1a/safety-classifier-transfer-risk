"""Nodes for the attacks pipeline (SPEC.md §8).

TextAttack runs in-process in the main environment via the ``turn1a/TextAttack`` fork
(transformers>=5 compatible). The sweep parallelises the independent
``(surrogate, recipe, shard)`` units across CPU-core worker processes: the per-example
search is CPU-forward-bound on these small models (MPS gives no speedup and a single MPS
device cannot be parallelised), so cores are the lever. Sharding a cell's examples means a
slow ``(surrogate, recipe)`` cell spreads across cores instead of pinning one worker.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _shard_indices(n: int, shard_size: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` spans of at most ``shard_size`` covering ``range(n)``.

    Args:
        n: number of examples to cover.
        shard_size: maximum examples per span (coerced to ``>= 1``).

    Returns:
        Ordered ``(start, stop)`` spans whose union is ``range(n)`` with no gaps or overlaps.
    """
    step = max(1, shard_size)
    return [(start, min(start + step, n)) for start in range(0, n, step)]


def run_attacks(
    splits: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
    params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Run each recipe against every surrogate, parallelised across CPU cores.

    The whole pool is attacked (not just the CKA-selected M1/M2) so the risk stage's
    ablation can compare the guided subset against random subsets drawn from all of them.
    Each ``(surrogate, recipe, shard)`` is an independent task dispatched to a worker
    process; a cell's per-example records are reassembled in original order afterwards.

    Args:
        splits: train/val/test DataFrames; the eval set is drawn from ``test``.
        manifest: surrogate manifest, ``name -> {"kind", "source", ...}``.
        params: the ``attacks`` block (recipes, eval_set_size, query_budget,
            max_prompt_chars, shard_size, semantic_encoder, num_workers).
        seed: root seed forwarded to TextAttack for reproducible sampling.

    Returns:
        Mapping ``"<surrogate>__<recipe>" -> [record, ...]`` of adversarial examples.
    """
    # Set these before importing the runner: the fork binds its device at import time, and
    # the workers (spawned) inherit this env. CPU + single-threaded so N workers use N cores.
    os.environ["TA_DEVICE"] = "cpu"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TA_SENTENCE_ENCODER"] = str(params.get("semantic_encoder", "sentence-transformers"))
    from transfer_risk.pipelines.attacks.runner import attack_shard  # noqa: PLC0415

    eval_size = int(params["eval_set_size"])
    query_budget = int(params["query_budget"])
    max_chars = int(params.get("max_prompt_chars") or 10**9)
    recipes = params["recipes"]
    test_df = splits["test"]
    injections = test_df.loc[test_df["label"] == 1, "text"].head(eval_size).tolist()
    # Truncate before attacking: the greedy word-level search cost scales with the prompt's
    # word count, and the prompts are long-tailed (parameters_attacks.yml). Injections are
    # front-loaded and the surrogates only see 256 tokens, so this bounds search cost without
    # changing the comparison (uniform across surrogates).
    examples = [{"text": text[:max_chars], "label": 1} for text in injections]
    # Shard each cell's examples across workers so a slow cell does not pin one core. The
    # default shard_size is the whole eval set (one shard per cell = pre-sharding behaviour).
    shard_size = int(params.get("shard_size") or len(examples)) or 1
    spans = _shard_indices(len(examples), shard_size)
    tasks = [
        (name, entry, recipe, start, stop)
        for name, entry in manifest.items()
        for recipe in recipes
        for start, stop in spans
    ]
    configured = params.get("num_workers")
    workers = int(configured) if configured else max(1, (os.cpu_count() or 2) - 2)
    workers = min(workers, len(tasks))
    logger.info(
        "Attacking %d cells in %d shards (<=%d ex) over %d examples on %d CPU workers",
        len(manifest) * len(recipes),
        len(tasks),
        shard_size,
        len(examples),
        workers,
    )
    cell_shards: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for name, entry, recipe, start, stop in tasks:
            future = pool.submit(
                attack_shard,
                entry,
                recipe,
                examples,
                start=start,
                stop=stop,
                query_budget=query_budget,
                seed=seed,
            )
            futures[future] = (name, recipe, start)
        for future in as_completed(futures):
            name, recipe, start = futures[future]
            records = future.result()
            for record in records:
                record["surrogate"] = name
                record["recipe"] = recipe
            cell_shards[f"{name}__{recipe}"][start] = records
    adversarial: dict[str, Any] = {}
    for cellkey, by_start in cell_shards.items():
        merged = [rec for start in sorted(by_start) for rec in by_start[start]]
        successes = sum(1 for rec in merged if rec["success"])
        logger.info("  %s succeeded on %d/%d", cellkey, successes, len(merged))
        adversarial[cellkey] = merged
    return adversarial
