"""Run-specific JSON catalog outputs for the prospective v2 protocol."""

from __future__ import annotations

from typing import Any, Literal

from kedro.io import AbstractDataset
from kedro_datasets.json import JSONDataset

from transfer_risk.datasets.experiment_v2_paths import current_project_root, guarded_run_path


class ExperimentV2JSONDataset(AbstractDataset[Any, Any]):
    """Keep relative and absolute catalog outputs within one local project/run namespace."""

    def __init__(
        self,
        *,
        filepath: str,
        run_id: str,
        visibility: Literal["private", "public"] = "private",
        save_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Bind an output to the declared local run-specific catalog root.

        Args:
            filepath: Relative or absolute filepath under this project's exact v2 run root.
            run_id: Dedicated lowercase run identity.
            visibility: Private evidence or public aggregate artifact root.
            save_args: JSON writer arguments.
            metadata: Optional Kedro dataset metadata.

        Raises:
            DatasetError: If the path is outside the exact isolated run root.
        """
        self._project_root = current_project_root()
        self._configured_path = guarded_run_path(
            filepath, project_root=self._project_root, run_id=run_id, visibility=visibility
        )
        # Kedro-MLflow discovers a custom dataset's local artifact through _filepath.
        self._filepath = self._configured_path
        self._run_id = run_id
        self._visibility = visibility
        self._assert_no_symlink()
        self._dataset = JSONDataset(
            filepath=str(self._configured_path),
            save_args={"indent": 2, "allow_nan": False, **(save_args or {})},
            metadata=metadata,
        )

    def _assert_no_symlink(self) -> None:
        """Reject redirection before catalog read or write operations.

        Raises:
            DatasetError: If any output component is a symlink or escapes the project.
        """
        guarded_run_path(
            self._configured_path,
            project_root=self._project_root,
            run_id=self._run_id,
            visibility=self._visibility,
        )

    def load(self) -> Any:
        """Read only after rechecking for output-path redirection.

        Returns:
            The parsed JSON evidence or aggregate artifact.
        """
        self._assert_no_symlink()
        return self._dataset.load()

    def save(self, data: Any) -> None:
        """Save only after rechecking for output-path redirection.

        Args:
            data: JSON-compatible evidence or public aggregate artifact.
        """
        self._assert_no_symlink()
        self._dataset.save(data)

    def exists(self) -> bool:
        """Check existence only after validating the isolated path.

        Returns:
            Whether the isolated JSON artifact exists.
        """
        self._assert_no_symlink()
        return bool(self._dataset.exists())

    def _describe(self) -> dict[str, Any]:
        """Describe the isolated catalog boundary without reading its content.

        Returns:
            Run identity, visibility, and output filepath.
        """
        return {
            "filepath": str(self._configured_path.relative_to(self._project_root)),
            "run_id": self._run_id,
            "visibility": self._visibility,
        }
