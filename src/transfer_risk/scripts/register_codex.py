"""Delegate both register passes to the installed subscription-authenticated reader."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def review_with_codex(
    subject: Path, *, label: str, tier: str, cache: Path, reader: Path | None = None
) -> int:
    """Run the installed gate with its qualified defaults and isolated local receipts.

    Args:
        subject: Markdown copy of authored prose or extracted SVG labels.
        label: Original subject path for diagnostics and the installed receipt.
        tier: Reader context passed to the installed gate.
        cache: Ignored local state directory; old custom receipts are never read.
        reader: Optional executable replacing the external gate in focused tests.

    Returns:
        The installed gate status, or two when the gate cannot be executed.
    """
    executable = reader or Path.home() / "dev/vl-branding/checks/register-pass.sh"
    environment = os.environ.copy()
    for name in ("REGISTER_MODEL", "REGISTER_EFFORT", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        environment.pop(name, None)
    state = cache.resolve() / "installed"
    environment.update(
        REGISTER_PROVIDER="codex",
        REGISTER_STATE_DIR=str(state),
        REGISTER_RECEIPTS=str(state / "receipts.tsv"),
    )
    try:
        state.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(  # noqa: S603
            [
                str(executable),
                "--provider",
                "codex",
                "--tier",
                tier,
                "--label",
                label,
                str(subject),
            ],
            env=environment,
            check=False,
        )
    except OSError as error:
        logger.error("Installed register gate unavailable for %s: %s", label, error)
        return 2
    return result.returncode if result.returncode in {0, 1, 2} else 2
