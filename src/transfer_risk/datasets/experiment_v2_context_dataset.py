"""Read actual local source and hardware provenance before prospective execution."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset, DatasetError


def git_output(*arguments: str) -> str:
    """Read one local Git fact without mutating repository state.

    Args:
        arguments: Read-only Git arguments selected by the dataset implementation.

    Returns:
        Stripped UTF-8 command output.
    """
    return subprocess.run(  # noqa: S603
        ["git", *arguments],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class ExperimentV2ExecutionContextDataset(AbstractDataset[Any, dict[str, Any]]):
    """Expose live worktree isolation, branch, clean source, commit, and host facts."""

    def __init__(self, *, metadata: dict[str, Any] | None = None) -> None:
        """Store optional catalog metadata without running any command.

        Args:
            metadata: Optional Kedro metadata.
        """
        self.metadata = metadata

    def load(self) -> dict[str, Any]:
        """Read current facts once per execution gate.

        Returns:
            Actual checkout and source state plus host/Python provenance.
        """
        root = Path(git_output("rev-parse", "--show-toplevel")).resolve()
        worktrees = git_output("worktree", "list", "--porcelain")
        primary = next(line[9:] for line in worktrees.splitlines() if line.startswith("worktree "))
        return {
            "project_root": str(root),
            "branch": git_output("branch", "--show-current"),
            "execution_ref": git_output("rev-parse", "HEAD"),
            "source_clean": not git_output("status", "--porcelain", "--untracked-files=normal"),
            "isolated_worktree": root != Path(primary).resolve(),
            "hardware": (
                f"{platform.platform()}; machine={platform.machine()}; "
                f"Python={sys.version.split()[0]}"
            ),
        }

    def save(self, data: Any) -> None:  # noqa: ARG002
        """Reject writes to observed source/hardware state.

        Args:
            data: Unsupported write value.

        Raises:
            DatasetError: Always, because execution context is observed read-only state.
        """
        msg = "v2 execution context is read-only"
        raise DatasetError(msg)

    def _describe(self) -> dict[str, str]:
        """Describe the local read-only provenance boundary.

        Returns:
            The context source.
        """
        return {"source": "live local Git checkout and host"}
