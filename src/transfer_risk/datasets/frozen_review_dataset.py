"""Private immutable local JSON storage for frozen review cohorts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset, DatasetError


class FrozenReviewDataset(AbstractDataset[dict[str, Any], dict[str, Any]]):
    """Save a private cohort once and accept only identical subsequent saves."""

    def __init__(self, *, filepath: str) -> None:
        """Store the catalog-owned local path without creating files."""
        self._path = Path(filepath)

    def _load(self) -> dict[str, Any]:
        """Load the frozen private JSON object."""
        value: dict[str, Any] = json.loads(self._path.read_text(encoding="utf-8"))
        return value

    def _save(self, data: dict[str, Any]) -> None:
        """Refuse replacement and create a new cohort exclusively when absent."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            if self._load() != data:
                msg = "refusing to replace a different frozen review cohort"
                raise DatasetError(msg)
            return
        with self._path.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)

    def _exists(self) -> bool:
        """Report whether the private frozen output already exists."""
        return self._path.exists()

    def _describe(self) -> dict[str, Any]:
        """Describe the catalog path without exposing cohort data."""
        return {"filepath": str(self._path)}
