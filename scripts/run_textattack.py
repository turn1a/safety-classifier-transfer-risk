"""Run one TextAttack recipe in an isolated subenv (Python 3.11 + transformers<5).

This script is invoked as a subprocess by the attacks node
(``pipelines/attacks/nodes.py``) and is NEVER imported by the main package. TextAttack
0.3.10 installs on the 2026 stack but does not run there: its ``flair`` dependency
imports ``TransfoXLTokenizer``, which transformers 5.x removed. So the attack sweep
runs here against a pinned older stack (resolved on the fly by ``uv run --no-project``).

Contract: read a JSONL eval set ({"text", "label"}) and a model spec (kind + source),
run the recipe, and write adversarial results as JSONL with one object per attacked
example ({"original", "perturbed", "original_label", "success", "result_type"}).

It is intentionally self-contained (it re-declares a minimal BiLSTM) because the subenv
does not have the project installed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from textattack import AttackArgs, Attacker
from textattack.attack_recipes import (
    BAEGarg2019,
    BERTAttackLi2020,
    DeepWordBugGao2018,
    PWWSRen2019,
    TextFoolerJin2019,
)
from textattack.datasets import Dataset
from textattack.models.wrappers import HuggingFaceModelWrapper, ModelWrapper

RECIPES = {
    "deepwordbug": DeepWordBugGao2018,
    "bert-attack": BERTAttackLi2020,
    "bae": BAEGarg2019,
    "pwws": PWWSRen2019,
    "textfooler": TextFoolerJin2019,
}

_TOKEN = re.compile(r"\w+")
PAD_ID = 0
UNK_ID = 1


def _tokenize(text):
    return _TOKEN.findall(text.lower())


def _encode(text, vocab, max_len):
    ids = [vocab.get(tok, UNK_ID) for tok in _tokenize(text)[:max_len]]
    return ids or [UNK_ID]


class _BiLSTM(torch.nn.Module):
    """Minimal mirror of transfer_risk.pipelines.models.bilstm.BiLSTMClassifier."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = torch.nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = torch.nn.Linear(2 * hidden_dim, 1)
        self.classifier = torch.nn.Linear(2 * hidden_dim, 2)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        weights = torch.softmax(self.attention(lstm_out), dim=1)
        return self.classifier((lstm_out * weights).sum(dim=1))


class _BiLSTMWrapper(ModelWrapper):
    """TextAttack wrapper around the from-scratch BiLSTM checkpoint."""

    def __init__(self, model_dir):
        config = json.loads((Path(model_dir) / "config.json").read_text())
        self.vocab = config["vocab"]
        self.max_len = config["max_seq_len"]
        self.model = _BiLSTM(
            len(self.vocab),
            config["embed_dim"],
            config["hidden_dim"],
            config["num_layers"],
            config["dropout"],
        )
        self.model.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location="cpu"))
        self.model.eval()

    def __call__(self, text_list):
        sequences = [_encode(text, self.vocab, self.max_len) for text in text_list]
        max_len = max(len(seq) for seq in sequences)
        padded = torch.tensor(
            [seq + [PAD_ID] * (max_len - len(seq)) for seq in sequences], dtype=torch.long
        )
        with torch.no_grad():
            return torch.softmax(self.model(padded), dim=-1).numpy()


def _load_wrapper(kind, source):
    if kind == "bilstm":
        return _BiLSTMWrapper(source)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)
    return HuggingFaceModelWrapper(model, tokenizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-budget", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    wrapper = _load_wrapper(args.kind, args.source)
    examples = [
        json.loads(line)
        for line in Path(args.input).read_text().splitlines()
        if line.strip()
    ]
    dataset = Dataset([(ex["text"], int(ex["label"])) for ex in examples])

    attack = RECIPES[args.recipe].build(wrapper)
    attack.goal_function.query_budget = args.query_budget

    attack_args = AttackArgs(
        num_examples=len(examples),
        disable_stdout=True,
        silent=True,
        random_seed=args.seed,
    )
    results = Attacker(attack, dataset, attack_args).attack_dataset()

    records = []
    for result in results:
        records.append(
            {
                "original": result.original_result.attacked_text.text,
                "perturbed": result.perturbed_result.attacked_text.text,
                "original_label": int(result.original_result.ground_truth_output),
                "success": result.__class__.__name__ == "SuccessfulAttackResult",
                "result_type": result.__class__.__name__,
            }
        )
    Path(args.output).write_text("\n".join(json.dumps(record) for record in records))


if __name__ == "__main__":
    main()
