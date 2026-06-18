"""A Kedro dataset for a trained surrogate classifier checkpoint.

kedro-datasets has no class for a fine-tuned HuggingFace sequence classifier (with its
tokenizer) or the project's from-scratch BiLSTM, so this dataset owns that serialisation in
one tested place, replacing the ``save_pretrained`` / ``torch.save`` calls that were otherwise
scattered across the training code. It is registered as the ``surrogate.{name}`` factory, so
every checkpoint — pretrained, fine-tuned, or BiLSTM — is a first-class catalog entry. Built on
:class:`~transfer_risk.datasets._fsspec_model_dir.FsspecModelDirDataset`, it works on local
paths and ``s3://`` alike; ``load`` returns the local directory path (materialising from S3 if
needed) and the caller reconstructs the model with the options it needs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import torch

from transfer_risk.datasets._fsspec_model_dir import FsspecModelDirDataset

if TYPE_CHECKING:
    from pathlib import Path

_CONFIG = "config.json"
_STATE = "model.pt"


class SurrogateModelDataset(FsspecModelDirDataset):
    """Persist a surrogate checkpoint (HuggingFace classifier or BiLSTM) to a directory.

    The saved value is a "bundle" dict. A transformer bundle is
    ``{"kind": "transformer", "model": <PreTrainedModel>, "tokenizer": <tokenizer>}``; a BiLSTM
    bundle is ``{"kind": "bilstm", "model": <BiLSTMClassifier>, "config": {...}}`` where
    ``config`` carries the vocab and layer sizes. ``load`` returns the local directory path; the
    on-disk layout (a ``config.json`` with a ``vocab`` is the BiLSTM, otherwise a HuggingFace
    directory) lets the consumer infer the kind.
    """

    def _write_dir(self, data: Any, dest: Path) -> None:
        """Write a transformer (``save_pretrained``) or BiLSTM (state dict + config) bundle."""
        if data["kind"] == "bilstm":
            torch.save(data["model"].state_dict(), dest / _STATE)
            (dest / _CONFIG).write_text(json.dumps(data["config"]))
        else:
            data["model"].save_pretrained(dest)
            data["tokenizer"].save_pretrained(dest)
