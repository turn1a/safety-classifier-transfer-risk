"""Run the project via ``python -m transfer_risk`` or the ``transfer-risk`` script.

This dispatches to the Kedro project CLI, so ``transfer-risk run --pipeline smoke``
is equivalent to ``kedro run --pipeline smoke``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from kedro.framework.cli.utils import find_run_command
from kedro.framework.project import configure_project

if TYPE_CHECKING:
    from typing import Any


def main(*args: Any, **kwargs: Any) -> Any:
    """Configure the project and dispatch to the Kedro project CLI."""
    package_name = Path(__file__).parent.name
    configure_project(package_name)
    kwargs["standalone_mode"] = not hasattr(sys, "ps1")
    run = find_run_command(package_name)
    return run(*args, **kwargs)


if __name__ == "__main__":
    main()
