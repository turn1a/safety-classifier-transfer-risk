"""Nodes for the attacks pipeline (SPEC.md §8).

TextAttack runs in-process in the main environment via the ``turn1a/TextAttack`` fork
(transformers>=5 compatible). The sweep parallelises the independent ``(surrogate, recipe)``
attacks across CPU-core worker processes: the per-example search is CPU-forward-bound on
these small models (MPS gives no speedup and a single MPS device cannot be parallelised),
so cores are the lever — roughly an N-times speedup on N cores, with no change to any
recipe's behaviour.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def run_attacks(
    splits: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
    params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Run each recipe against every surrogate, parallelised across CPU cores.

    The whole pool is attacked (not just the CKA-selected M1/M2) so the risk stage's
    ablation can compare the guided subset against random subsets drawn from all of them.
    Each ``(surrogate, recipe)`` pair is an independent task dispatched to a worker process.

    Args:
        splits: train/val/test DataFrames; the eval set is drawn from ``test``.
        manifest: surrogate manifest, ``name -> {"kind", "source", ...}``.
        params: the ``attacks`` block (recipes, eval_set_size, query_budget,
            semantic_encoder, num_workers).
        seed: root seed forwarded to TextAttack for reproducible sampling.

    Returns:
        Mapping ``"<surrogate>__<recipe>" -> [record, ...]`` of adversarial examples.
    """
    # Set these before importing the runner: the fork binds its device at import time, and
    # the workers (spawned) inherit this env. CPU + single-threaded so N workers use N cores.
    os.environ["TA_DEVICE"] = "cpu"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TA_SENTENCE_ENCODER"] = str(params.get("semantic_encoder", "sentence-transformers"))
    from transfer_risk.pipelines.attacks.runner import attack_one  # noqa: PLC0415

    eval_size = int(params["eval_set_size"])
    query_budget = int(params["query_budget"])
    max_chars = int(params["max_prompt_chars"])
    recipes = params["recipes"]
    test_df = splits["test"]
    injections = test_df.loc[test_df["label"] == 1, "text"].head(eval_size).tolist()
    # Truncate before attacking: the greedy word-level search cost scales with the prompt's
    # word count, and the prompts are long-tailed (params_attacks.yml). Injections are
    # front-loaded and the surrogates only see 256 tokens, so this bounds search cost without
    # changing the comparison (uniform across surrogates).
    examples = [{"text": text[:max_chars], "label": 1} for text in injections]
    tasks = [(name, entry, recipe) for name, entry in manifest.items() for recipe in recipes]
    configured = params.get("num_workers")
    workers = int(configured) if configured else max(1, (os.cpu_count() or 2) - 2)
    workers = min(workers, len(tasks))
    logger.info(
        "Attacking %d (surrogate x recipe) tasks over %d examples on %d CPU workers",
        len(tasks),
        len(examples),
        workers,
    )
    adversarial: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for name, entry, recipe in tasks:
            future = pool.submit(
                attack_one, entry, recipe, examples, query_budget=query_budget, seed=seed
            )
            futures[future] = (name, recipe)
        for future in as_completed(futures):
            name, recipe = futures[future]
            records = future.result()
            for record in records:
                record["surrogate"] = name
                record["recipe"] = recipe
            successes = sum(1 for record in records if record["success"])
            logger.info("  %s/%s succeeded on %d/%d", name, recipe, successes, len(records))
            adversarial[f"{name}__{recipe}"] = records
    return adversarial
