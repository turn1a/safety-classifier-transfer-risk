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

import numpy as np
import onnxruntime as ort
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
from transformers import AutoTokenizer

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

# Candidate-query batch size for the goal function (default 32). The per-example greedy
# search is bandwidth-bound on CPU — each forward streams the full model weights once, so
# batching more candidates per forward amortises that stream. Larger batches cut wall time
# without changing any prediction (batching does not affect masked per-example logits).
_QUERY_BATCH_SIZE = 128


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


class ONNXModelWrapper(ModelWrapper):  # type: ignore[misc]  # ModelWrapper is untyped (Any)
    """TextAttack wrapper that serves a surrogate's exported ONNX graph via ONNX Runtime.

    Same ``text -> class-probability (n, 2)`` contract as the HF/BiLSTM wrappers, so the
    search is unchanged; only the victim forward is swapped for an ONNX Runtime session,
    which is ~2-3x faster per query on CPU. It imports *only* ``onnxruntime`` plus the
    project's tokenizer — never ``optimum`` — so the hot path avoids the
    optimum<->transformers-5 version conflict (the ``.onnx`` graph is produced offline by
    ``just export-onnx``). One intra-op thread, so N pool workers use N cores.
    """

    def __init__(self, onnx_path: str, tokenizer_source: str, *, max_seq_len: int = 256) -> None:
        """Open the ONNX session at ``onnx_path`` with the tokenizer at ``tokenizer_source``."""
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self.input_names = {model_input.name for model_input in self.session.get_inputs()}
        self.max_seq_len = max_seq_len

    def __call__(self, text_list: list[str]) -> Any:
        """Return softmax class probabilities ``(n, 2)`` for a list of texts."""
        encoded = self.tokenizer(
            text_list,
            truncation=True,
            max_length=self.max_seq_len,
            padding=True,
            return_tensors="np",
        )
        feeds = {name: encoded[name] for name in self.input_names if name in encoded}
        logits = self.session.run(None, feeds)[0]
        shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return shifted / shifted.sum(axis=-1, keepdims=True)


def build_wrapper(
    entry: Mapping[str, Any], device: torch.device, *, onnx_dir: str | None = None
) -> ModelWrapper:
    """Build a TextAttack ``ModelWrapper`` for one surrogate manifest entry.

    With ``onnx_dir`` set (and the surrogate not the BiLSTM), the victim is served from the
    exported graph at ``{onnx_dir}/model.onnx`` via :class:`ONNXModelWrapper`; otherwise the
    torch checkpoint is loaded. The BiLSTM is tiny and always stays on torch.
    """
    if entry["kind"] == "bilstm":
        return BiLSTMModelWrapper(entry["source"], device)
    if onnx_dir is not None:
        return ONNXModelWrapper(f"{onnx_dir}/model.onnx", onnx_dir)
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
    attack.goal_function.batch_size = _QUERY_BATCH_SIZE
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
