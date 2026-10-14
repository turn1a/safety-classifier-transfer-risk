"""Validate local v2 catalog paths against one captured project and run namespace."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from kedro.io import DatasetError
from kedro.utils import find_kedro_project


def current_project_root() -> Path:
    """Capture the canonical Kedro project root without inferring it from an output path.

    Returns:
        The current Kedro project, or the current directory for direct dataset use.
    """
    current = Path.cwd()
    return Path(find_kedro_project(current) or current).resolve()


def guarded_run_path(
    filepath: str | Path,
    *,
    project_root: Path,
    run_id: str,
    visibility: Literal["private", "public"] = "private",
) -> Path:
    """Accept relative or absolute paths only in this project's exact v2 run root.

    The file path remains lexical until its symlink components are checked.
    Resolving it first would hide a redirect to another directory.

    Args:
        filepath: Project-relative or Kedro-normalized absolute local path.
        project_root: Canonical project captured when the dataset was constructed.
        run_id: Declared lowercase run identity.
        visibility: Private data or public aggregate namespace.

    Returns:
        The checked absolute local path, independent of later working-directory changes.

    Raises:
        DatasetError: If the path leaves the declared project/run or traverses a symlink.
    """
    path = Path(filepath)
    prefix = "data" if visibility == "private" else "docs/artifacts"
    expected = project_root / prefix / "experiment_v2" / run_id
    configured = path if path.is_absolute() else project_root / path
    if (
        not re.fullmatch("[a-z][a-z0-9-]{3,63}", run_id)
        or visibility not in {"private", "public"}
        or "://" in str(filepath)
        or ".." in path.parts
        or configured == expected
        or not configured.is_relative_to(expected)
    ):
        message = "v2 catalog output must use its isolated local project and run-specific root"
        raise DatasetError(message)
    for component in (configured, *configured.parents):
        if component.is_symlink():
            message = "v2 catalog output cannot traverse a symlink"
            raise DatasetError(message)
    return configured
