"""Node for the smoke pipeline — the one implemented node in this scaffold."""

from __future__ import annotations

import platform
import sys
from typing import TYPE_CHECKING

import kedro

if TYPE_CHECKING:
    from typing import Any


def record_environment(project: dict[str, Any], seed: int) -> dict[str, Any]:
    """Record a small environment fingerprint.

    Writing this through the catalog (``smoke_report``) and running it under
    kedro-mlflow proves the catalog + tracking wiring without any heavy ML
    dependency.

    Args:
        project: The ``project`` parameter block (name, version).
        seed: The project root seed.

    Returns:
        A JSON-serialisable dict describing the runtime environment.
    """
    return {
        "status": "ok",
        "project": project.get("name", "unknown"),
        "version": project.get("version", "unknown"),
        "seed": seed,
        "kedro_version": kedro.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
