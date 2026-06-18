"""Nodes for the attacks pipeline (SPEC.md §8).

TextAttack runs in-process in the main environment via the ``turn1a/TextAttack`` fork
(transformers>=5 compatible). The sweep parallelises the independent
``(surrogate, recipe, shard)`` units across CPU-core worker processes: the per-example
search is CPU-forward-bound on these small models (MPS gives no speedup and a single MPS
device cannot be parallelised), so cores are the lever. Sharding a cell's examples means a
slow ``(surrogate, recipe)`` cell spreads across cores instead of pinning one worker. Each
completed cell is written immediately as a partition file and skipped on a re-run, so an
interrupted sweep (e.g. a reclaimed spot instance) resumes where it left off.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from kedro_datasets.pickle import PickleDataset

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


def _existing_cells(root: str) -> set[str]:
    """Cell ids already written under ``root`` (so a re-run skips them); empty if none yet.

    The directory is read directly rather than as a catalog input: Kedro's
    ``PartitionedDataset`` raises if a directory has no partitions, which is exactly the
    first-run state, so the node lists it itself and treats "missing or empty" as "nothing
    done yet". ``root`` is a local path; cross-machine resume is handled out of band by the
    cloud box's ``aws s3 sync`` of this directory, not by Kedro reading S3.

    Args:
        root: the partition directory (``attacks.partition_root``).

    Returns:
        The set of ``"<surrogate>__<recipe>"`` ids that already have a ``.pkl`` partition.
    """
    return {path.stem for path in Path(root).glob("*.pkl")}


def _save_partition(root: str, cellkey: str, records: list[dict[str, Any]]) -> None:
    """Persist one cell's records to ``{root}/{cellkey}.pkl`` as it completes (resume durability).

    Writing each cell the moment its shards finish — rather than only when the whole node
    returns — bounds the loss from an interruption to the in-flight cells, and is what the
    cloud box's periodic ``s3 sync`` of the partition directory picks up. ``root`` matches the
    ``adversarial_examples`` PartitionedDataset path so the same files are read back on a
    re-run and consumed downstream.

    Args:
        root: the partition directory (``attacks.partition_root``).
        cellkey: the ``"<surrogate>__<recipe>"`` partition id.
        records: the cell's per-example records.
    """
    PickleDataset(filepath=f"{root}/{cellkey}.pkl").save(records)


def run_attacks(
    splits: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
    params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Run each recipe against every surrogate, parallelised and resumable across CPU cores.

    The whole pool is attacked (not just the CKA-selected M1/M2) so the risk stage's
    ablation can compare the guided subset against random subsets drawn from all of them.
    Each ``(surrogate, recipe, shard)`` is an independent task dispatched to a worker
    process; a cell's per-example records are reassembled in original order and written as a
    partition the moment its shards complete. Cells already on disk under
    ``params["partition_root"]`` are skipped, so a re-run resumes.

    Args:
        splits: train/val/test DataFrames; the eval set is drawn from ``test``.
        manifest: surrogate manifest, ``name -> {"kind", "source", ...}``.
        params: the ``attacks`` block (recipes, eval_set_size, query_budget,
            max_prompt_chars, shard_size, partition_root, semantic_encoder, num_workers).
        seed: root seed forwarded to TextAttack for reproducible sampling.

    Returns:
        Mapping ``"<surrogate>__<recipe>" -> [record, ...]`` of the cells computed this run
        (skipped cells stay on disk untouched).
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
    partition_root = str(params["partition_root"])
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
    # Resume: skip cells already persisted by a previous (possibly interrupted) run.
    done = _existing_cells(partition_root)
    tasks = [
        (name, entry, recipe, start, stop)
        for name, entry in manifest.items()
        for recipe in recipes
        for start, stop in spans
        if f"{name}__{recipe}" not in done
    ]
    cells_total = len(manifest) * len(recipes)
    if not tasks:
        logger.info("All %d cells already present; nothing to attack", cells_total)
        return {}
    configured = params.get("num_workers")
    workers = int(configured) if configured else max(1, (os.cpu_count() or 2) - 2)
    workers = min(workers, len(tasks))
    logger.info(
        "Attacking %d of %d cells in %d shards (<=%d ex) over %d examples on %d CPU workers",
        cells_total - len(done),
        cells_total,
        len(tasks),
        shard_size,
        len(examples),
        workers,
    )
    shards_per_cell = len(spans)
    pending: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    new_partitions: dict[str, Any] = {}
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
            cellkey = f"{name}__{recipe}"
            pending[cellkey][start] = records
            if len(pending[cellkey]) == shards_per_cell:
                by_start = pending.pop(cellkey)
                merged = [rec for shard_start in sorted(by_start) for rec in by_start[shard_start]]
                _save_partition(partition_root, cellkey, merged)
                successes = sum(1 for rec in merged if rec["success"])
                logger.info("  %s succeeded on %d/%d (saved)", cellkey, successes, len(merged))
                new_partitions[cellkey] = merged
    return new_partitions
