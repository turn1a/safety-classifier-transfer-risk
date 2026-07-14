"""Sanitize local paths from a static Kedro-viz export."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
_LOCAL_USER_PATH = re.compile(r"/Users/[A-Za-z0-9._-]+(?:/[^\s\"'<>\\\r\n]*)?")


def sanitize_viz_text(text: str, workspace_root: Path) -> str:
    """Replace workspace and other macOS user paths in generated visualization text.

    Args:
        text: One UTF-8 static-site or API payload.
        workspace_root: Absolute repository workspace path to redact first.

    Returns:
        Text with local workspace and macOS user paths replaced by neutral placeholders.
    """
    workspace = workspace_root.resolve().as_posix()
    return _LOCAL_USER_PATH.sub("<local-path>", text.replace(workspace, "<workspace>"))


def sanitize_viz_tree(root: Path, workspace_root: Path) -> int:
    """Sanitize every UTF-8 file below a generated static visualization directory.

    Args:
        root: Static Kedro-viz output directory.
        workspace_root: Absolute repository workspace path to redact.

    Returns:
        Number of text files whose contents changed.
    """
    changed = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        sanitized = sanitize_viz_text(text, workspace_root)
        if sanitized == text:
            continue
        path.write_text(sanitized, encoding="utf-8")
        changed += 1
    return changed


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse static visualization sanitation command-line arguments.

    Args:
        argv: Optional command-line arguments without the executable name.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Generated docs/pipeline-viz directory")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="Absolute local workspace path to redact",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Sanitize the generated static Kedro-viz bundle and return a process status.

    Args:
        argv: Optional command-line arguments without the executable name.

    Returns:
        Zero when a static visualization directory was sanitized.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    root = args.root
    if not root.is_dir():
        msg = f"Static visualization directory does not exist: {root}"
        raise ValueError(msg)
    changed = sanitize_viz_tree(root, args.workspace_root)
    logger.info("Sanitized %d generated visualization text file(s)", changed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
