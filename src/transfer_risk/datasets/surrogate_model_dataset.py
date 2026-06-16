"""A Kedro dataset for a trained surrogate classifier checkpoint.

kedro-datasets has no class for a fine-tuned HuggingFace sequence classifier (with its
tokenizer) or the project's from-scratch BiLSTM, so this dataset owns that serialization
in one tested place, replacing the ``save_pretrained`` / ``torch.save`` calls that were
otherwise scattered across the training and loading code. It is registered in the catalog
as the ``surrogate.{name}`` factory, so every checkpoint is a first-class catalog entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from kedro.io import AbstractDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_CONFIG = "config.json"
_STATE = "model.pt"


class SurrogateModelDataset(AbstractDataset[dict[str, Any], dict[str, Any]]):
    """Persist and restore a surrogate checkpoint (HuggingFace classifier or BiLSTM).

    The value is a "bundle" dict. A transformer bundle is
    ``{"kind": "transformer", "model": <PreTrainedModel>, "tokenizer": <tokenizer>}``; a
    BiLSTM bundle is ``{"kind": "bilstm", "model": <BiLSTMClassifier>, "config": {...}}``,
    where ``config`` carries the vocab and layer sizes. On load the kind is inferred from
    the on-disk layout: a ``config.json`` holding a ``vocab`` is the BiLSTM, otherwise it
    is a HuggingFace directory.
    """

    def __init__(self, filepath: str) -> None:
        """Store the checkpoint directory ``filepath``."""
        self._path = Path(filepath)

    def save(self, data: dict[str, Any]) -> None:
        """Write a transformer (``save_pretrained``) or BiLSTM (state dict + config) bundle."""
        self._path.mkdir(parents=True, exist_ok=True)
        if data["kind"] == "bilstm":
            torch.save(data["model"].state_dict(), self._path / _STATE)
            (self._path / _CONFIG).write_text(json.dumps(data["config"]))
        else:
            data["model"].save_pretrained(self._path)
            data["tokenizer"].save_pretrained(self._path)

    def load(self) -> dict[str, Any]:
        """Reconstruct the bundle, inferring the kind from the on-disk layout."""
        config_path = self._path / _CONFIG
        if (self._path / _STATE).exists() and config_path.exists():
            config = json.loads(config_path.read_text())
            if "vocab" in config:
                # Imported lazily: pipelines.models.__init__ eagerly loads the pipeline,
                # which imports training, which imports this dataset (a cycle at top level).
                from transfer_risk.pipelines.models.bilstm import (  # noqa: PLC0415
                    BiLSTMClassifier,
                )

                model = BiLSTMClassifier(
                    len(config["vocab"]),
                    config["embed_dim"],
                    config["hidden_dim"],
                    config["num_layers"],
                    config["dropout"],
                )
                model.load_state_dict(torch.load(self._path / _STATE, map_location="cpu"))
                model.eval()
                return {"kind": "bilstm", "model": model, "config": config}
        source = str(self._path)
        return {
            "kind": "transformer",
            "model": AutoModelForSequenceClassification.from_pretrained(source),
            "tokenizer": AutoTokenizer.from_pretrained(source),
        }

    def _exists(self) -> bool:
        """Whether the checkpoint directory has been written."""
        return self._path.exists() and any(self._path.iterdir())

    def _describe(self) -> dict[str, Any]:
        """Return a printable description of this dataset."""
        return {"filepath": str(self._path)}
