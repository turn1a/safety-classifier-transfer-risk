"""Read pinned local computational-screening models through the catalog."""

from __future__ import annotations

import re
from typing import Any

import torch
from kedro.io.core import AbstractDataset, DatasetError


class PinnedSemanticModelDataset(AbstractDataset[Any, dict[str, Any]]):
    """Load a text encoder or NLI classifier with the same pinned tokenizer revision."""

    def __init__(
        self,
        *,
        repo_id: str,
        revision: str,
        role: str,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        """Validate immutable screening identities before any model I/O.

        Args:
            repo_id: Public model repository.
            revision: Full 40-character immutable Hub commit.
            role: ``encoder`` or ``nli``; arbitrary model code is disabled.
            cache_dir: Optional task-owned Hub cache directory.
            local_files_only: Require cached assets instead of downloading them.

        Raises:
            ValueError: If a revision or model role is invalid.
        """
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            msg = "semantic models require a 40-character immutable Hub revision"
            raise ValueError(msg)
        if role not in {"encoder", "nli"}:
            msg = "semantic model role must be encoder or nli"
            raise ValueError(msg)
        self._repo_id, self._revision, self._role = repo_id, revision, role
        self._cache_dir, self._local_files_only = cache_dir, local_files_only

    def load(self) -> dict[str, Any]:
        """Return frozen FP32 model/tokenizer assets without remote code or pickle weights."""
        from transformers import (  # noqa: PLC0415 (defer heavy catalog import)
            AutoModel,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        shared = {
            "revision": self._revision,
            "cache_dir": self._cache_dir,
            "local_files_only": self._local_files_only,
            "trust_remote_code": False,
        }
        loader = AutoModel if self._role == "encoder" else AutoModelForSequenceClassification
        model = loader.from_pretrained(
            self._repo_id, dtype=torch.float32, use_safetensors=True, **shared
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(self._repo_id, **shared)
        return {
            "model": model,
            "tokenizer": tokenizer,
            "identity": {"repo_id": self._repo_id, "revision": self._revision, "role": self._role},
        }

    def save(self, data: Any) -> None:  # noqa: ARG002 (read-only catalog interface)
        """Reject writes to a read-only upstream model dataset."""
        msg = "pinned semantic model dataset is read-only"
        raise DatasetError(msg)

    def _describe(self) -> dict[str, Any]:
        """Describe public model identities without credentials or local text."""
        return {"repo_id": self._repo_id, "revision": self._revision, "role": self._role}
