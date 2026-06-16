"""Nodes for the attacks pipeline (SPEC.md §8).

TextAttack runs in-process in the main environment via the ``turn1a/TextAttack`` fork
(transformers>=5 compatible). This node loads each surrogate once and runs every
configured recipe against it, collecting adversarial examples for the transfer stage.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def run_attacks(
    splits: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
    params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Run each recipe against every surrogate in-process and collect adversarial examples.

    The whole pool is attacked (not just the CKA-selected M1/M2) so the risk stage's
    ablation can compare the guided subset against random subsets drawn from all of them;
    attacking only the selected set would make that comparison degenerate.

    Args:
        splits: train/val/test DataFrames; the eval set is drawn from ``test``.
        manifest: surrogate manifest, ``name -> {"kind", "source", ...}``.
        params: the ``attacks`` parameter block (recipes, eval_set_size, query_budget,
            semantic_encoder).
        seed: root seed forwarded to TextAttack for reproducible sampling.

    Returns:
        Mapping ``"<surrogate>__<recipe>" -> [record, ...]`` of adversarial examples.
    """
    # Import the TextAttack glue lazily so assembling the pipeline (and the registry
    # test) never imports textattack; the heavy import happens only when attacks run.
    from transfer_risk.pipelines.attacks.runner import build_wrapper, run_recipe  # noqa: PLC0415

    os.environ["TA_SENTENCE_ENCODER"] = str(params.get("semantic_encoder", "sentence-transformers"))
    eval_size = int(params["eval_set_size"])
    query_budget = int(params["query_budget"])
    recipes = params["recipes"]
    device = torch.device("cpu")  # attacks are query-bound forward passes; CPU is reliable
    test_df = splits["test"]
    injections = test_df.loc[test_df["label"] == 1, "text"].head(eval_size).tolist()
    examples = [{"text": text, "label": 1} for text in injections]
    adversarial: dict[str, Any] = {}
    for name, entry in manifest.items():
        wrapper = build_wrapper(entry, device)
        for recipe in recipes:
            logger.info("Attacking %s with %s over %d examples", name, recipe, len(examples))
            records = run_recipe(wrapper, recipe, examples, query_budget=query_budget, seed=seed)
            for record in records:
                record["surrogate"] = name
                record["recipe"] = recipe
            successes = sum(1 for record in records if record["success"])
            logger.info("  %s/%s succeeded on %d/%d", name, recipe, successes, len(records))
            adversarial[f"{name}__{recipe}"] = records
    return adversarial
