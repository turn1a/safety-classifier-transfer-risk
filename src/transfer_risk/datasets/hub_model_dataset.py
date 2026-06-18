"""A read-only Kedro dataset for a HuggingFace Hub sequence-classification model.

Pretrained surrogates, fine-tune backbones, and the frozen target are all hub models. Listing
them as catalog datasets (alongside the raw HF *dataset* entries) keeps the hub the catalog's
concern, never a ``from_pretrained`` call inside a node: a node takes this dataset's loaded
bundle and either persists it as a ``surrogate.{name}`` checkpoint, fine-tunes it, or reads
its representations. ``load_args`` per entry select the right load (``num_labels: 2`` for a
backbone, ``output_hidden_states: true`` for the target); the HF token, if needed for a gated
repo, comes from ``credentials`` or the ``HF_TOKEN`` environment variable.
"""

from __future__ import annotations

from typing import Any

import torch
from kedro.io.core import AbstractDataset, DatasetError


class HuggingFaceHubModelDataset(AbstractDataset[Any, dict[str, Any]]):
    """Load a HuggingFace classifier + tokenizer from the Hub as a reusable bundle.

    ``load`` returns ``{"kind": "transformer", "model": <PreTrainedModel>, "tokenizer": ...}``,
    loaded in fp32 by default (matching the project's training/inference path) with any
    ``load_args`` applied. The dataset is read-only: ``save`` raises, since the Hub is upstream.
    """

    def __init__(
        self,
        *,
        repo_id: str,
        load_args: dict[str, Any] | None = None,
        credentials: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store the Hub ``repo_id`` and load options.

        Args:
            repo_id: the HuggingFace Hub model id (e.g. ``protectai/deberta-...``).
            load_args: keyword arguments for ``from_pretrained`` (e.g. ``num_labels``,
                ``output_hidden_states``); ``dtype`` defaults to ``float32``.
            credentials: may carry a ``token`` for gated repos; falls back to ``HF_TOKEN``.
            metadata: arbitrary metadata, ignored by Kedro.
        """
        self._repo_id = repo_id
        self._load_args = dict(load_args or {})
        self._token = (credentials or {}).get("token")
        self.metadata = metadata

    def load(self) -> dict[str, Any]:
        """Return a ``{"kind": "transformer", "model", "tokenizer"}`` bundle from the Hub."""
        from transformers import (  # noqa: PLC0415 (heavy import, kept off catalog parse)
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        kwargs: dict[str, Any] = {"dtype": torch.float32, **self._load_args}
        model = AutoModelForSequenceClassification.from_pretrained(
            self._repo_id, token=self._token, **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(self._repo_id, token=self._token)
        return {"kind": "transformer", "model": model, "tokenizer": tokenizer}

    def save(self, data: Any) -> None:  # noqa: ARG002 (read-only dataset; arg is interface-only)
        """Raise: the Hub is an upstream source, so this dataset is read-only.

        Raises:
            DatasetError: always.
        """
        msg = f"{type(self).__name__} is read-only ({self._repo_id})"
        raise DatasetError(msg)

    def _describe(self) -> dict[str, Any]:
        """Return a printable description of this dataset."""
        return {"repo_id": self._repo_id, "load_args": self._load_args}
