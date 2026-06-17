"""In-process TextAttack runner (SPEC.md §8).

The attack sweep runs in the main environment via the ``turn1a/TextAttack`` fork
(transformers>=5 compatible). This module is the thin glue that imports ``textattack``:
it wraps a surrogate (a HuggingFace classifier or the project's ``BiLSTMClassifier``)
as a TextAttack ``ModelWrapper``, builds a recipe, runs the attack, and returns plain
records. It replaces the old ``scripts/run_textattack.py`` subprocess, which existed
only because TextAttack could not import under transformers 5.x.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from textattack import AttackArgs, Attacker
from textattack.attack_recipes import (
    BAEGarg2019,
    BERTAttackLi2020,
    DeepWordBugGao2018,
    PWWSRen2019,
    TextFoolerJin2019,
)
from textattack.datasets import Dataset as TextAttackDataset
from textattack.models.wrappers import HuggingFaceModelWrapper, ModelWrapper

from transfer_risk.modeling import load_transformer
from transfer_risk.pipelines.models.bilstm import BiLSTMClassifier, encode, pad_batch

if TYPE_CHECKING:
    from collections.abc import Mapping

RECIPES = {
    "deepwordbug": DeepWordBugGao2018,
    "pwws": PWWSRen2019,
    "textfooler": TextFoolerJin2019,
    "bae": BAEGarg2019,
    "bert-attack": BERTAttackLi2020,
}


class BiLSTMModelWrapper(ModelWrapper):  # type: ignore[misc]  # ModelWrapper is untyped (Any)
    """TextAttack wrapper around the project's saved ``BiLSTMClassifier`` checkpoint.

    Loads the from-scratch BiLSTM (its ``config.json`` carries the vocab and shape) and
    exposes the ``text -> class-probability`` callable TextAttack's goal function needs.
    Unlike the deleted subenv script, this wraps the real ``BiLSTMClassifier`` rather
    than re-declaring a copy, so the attacked model is byte-identical to the one trained
    and measured upstream.
    """

    def __init__(self, source: str, device: torch.device) -> None:
        """Load the BiLSTM checkpoint at ``source`` onto ``device`` in eval mode."""
        config = json.loads((Path(source) / "config.json").read_text())
        self.vocab = config["vocab"]
        self.max_len = config["max_seq_len"]
        self.device = device
        self.model = BiLSTMClassifier(
            len(self.vocab),
            config["embed_dim"],
            config["hidden_dim"],
            config["num_layers"],
            config["dropout"],
        )
        self.model.load_state_dict(torch.load(Path(source) / "model.pt", map_location=device))
        self.model.to(device).eval()

    def __call__(self, text_list: list[str]) -> Any:
        """Return class probabilities ``(n, 2)`` for a list of texts."""
        ids = pad_batch([encode(text, self.vocab, self.max_len) for text in text_list]).to(
            self.device
        )
        with torch.no_grad():
            return torch.softmax(self.model(ids), dim=-1).cpu().numpy()


def build_wrapper(entry: Mapping[str, Any], device: torch.device) -> ModelWrapper:
    """Build a TextAttack ``ModelWrapper`` for one surrogate manifest entry."""
    if entry["kind"] == "bilstm":
        return BiLSTMModelWrapper(entry["source"], device)
    model, tokenizer = load_transformer(entry["source"], device)
    return HuggingFaceModelWrapper(model, tokenizer)


def run_recipe(
    wrapper: ModelWrapper,
    recipe: str,
    examples: list[dict[str, Any]],
    *,
    query_budget: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run one recipe over ``examples`` and return one record per attacked example.

    Args:
        wrapper: The victim surrogate wrapped for TextAttack.
        recipe: A key of :data:`RECIPES`.
        examples: ``[{"text", "label"}]`` rows the surrogate should classify as injection.
        query_budget: Per-example cap on victim queries (SPEC.md §11).
        seed: Seed for TextAttack's sampling, for reproducibility.

    Returns:
        One dict per example with ``original``, ``perturbed``, ``original_label``,
        ``success`` and ``result_type``.
    """
    attack = RECIPES[recipe].build(wrapper)
    attack.goal_function.query_budget = query_budget
    dataset = TextAttackDataset([(ex["text"], int(ex["label"])) for ex in examples])
    attack_args = AttackArgs(
        num_examples=len(examples),
        disable_stdout=True,
        silent=True,
        random_seed=seed,
    )
    results = Attacker(attack, dataset, attack_args).attack_dataset()
    return [
        {
            "original": result.original_result.attacked_text.text,
            "perturbed": result.perturbed_result.attacked_text.text,
            "original_label": int(result.original_result.ground_truth_output),
            "success": result.__class__.__name__ == "SuccessfulAttackResult",
            "result_type": result.__class__.__name__,
        }
        for result in results
    ]


def attack_one(
    entry: Mapping[str, Any],
    recipe: str,
    examples: list[dict[str, Any]],
    *,
    query_budget: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run one ``(surrogate, recipe)`` attack task — the unit parallelised across workers.

    The attacks node submits one of these per ``(surrogate, recipe)`` to a process pool, so
    the embarrassingly-parallel sweep saturates the CPU cores. Each worker is pinned to a
    single torch thread so N workers use N cores without oversubscription; the victim and
    the masked-LM run on CPU (TextAttack's device is fixed to CPU via the inherited
    ``TA_DEVICE`` env before this module imports textattack). Loading the model inside the
    worker keeps the pickled payload to plain data, never a live model.

    Args:
        entry: the surrogate's manifest entry (``kind`` + ``source``).
        recipe: a key of :data:`RECIPES`.
        examples: the shared eval set (``[{"text", "label"}]``).
        query_budget: per-example victim-query cap.
        seed: TextAttack sampling seed.

    Returns:
        One record per attacked example (see :func:`run_recipe`).
    """
    torch.set_num_threads(1)
    wrapper = build_wrapper(entry, torch.device("cpu"))
    return run_recipe(wrapper, recipe, examples, query_budget=query_budget, seed=seed)
