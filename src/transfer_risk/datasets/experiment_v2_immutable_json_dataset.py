"""Prevent overwriting private v2 heavy-stage evidence after its first successful save."""

from __future__ import annotations

from typing import Any

from kedro.io import DatasetError

from transfer_risk.datasets.experiment_v2_json_dataset import ExperimentV2JSONDataset


class ExperimentV2ImmutableJSONDataset(ExperimentV2JSONDataset):
    """Retain original model, attempt, audit, and raw-record evidence for one run."""

    def save(self, data: Any) -> None:
        """Write fresh evidence while refusing to replace an existing run artifact.

        Args:
            data: JSON-compatible private evidence.

        Raises:
            DatasetError: If an immutable artifact already exists.
        """
        if self.exists():
            message = (
                "v2 heavy-stage evidence is immutable; resume missing outputs or use a fresh run"
            )
            raise DatasetError(message)
        self._publish(data, immutable=True)
