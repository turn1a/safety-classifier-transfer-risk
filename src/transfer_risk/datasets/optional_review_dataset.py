"""Catalog boundary for an intentionally absent finalized public review summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset


class OptionalReviewDataset(AbstractDataset[dict[str, Any] | None, dict[str, Any]]):
    """Load missing local evidence as None while refusing malformed existing evidence."""

    def __init__(self, *, filepath: str) -> None:
        """Retain the catalog-owned local summary path."""
        self._path = Path(filepath)

    def _load(self) -> dict[str, Any] | None:
        """Distinguish an absent artifact from invalid or unreadable present JSON."""
        try:
            content = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        value = json.loads(content)
        if not isinstance(value, dict):
            msg = "finalized review evidence must be a JSON object"
            raise ValueError(msg)
        return value

    def _save(self, data: dict[str, Any]) -> None:
        """Save an explicitly finalized summary through the existing catalog interface."""
        content = json.dumps(data, allow_nan=False, ensure_ascii=False, indent=2)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(content + "\n", encoding="utf-8")

    def _exists(self) -> bool:
        """Report actual persisted evidence, preserving Kedro resume behavior."""
        return self._path.exists()

    def _describe(self) -> dict[str, Any]:
        """Describe only the catalog path."""
        return {"filepath": str(self._path)}
