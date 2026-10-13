"""Run both register passes over authored prose, including Quarto source.

The branding repository supplies language checks only. Diagram labels are extracted as
text so personal-project figures are independent of its visual branding rules.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from transfer_risk.scripts.register_codex import review_with_codex

if TYPE_CHECKING:
    from collections.abc import Sequence

_PROSE_SUFFIXES = {".md", ".qmd", ".txt", ".svg"}
_EXCLUDED_ROOTS = {"docs/pipeline-viz", "docs/_site", "docs/.quarto"}
logger = logging.getLogger(__name__)


def _diagram_labels(path: Path) -> str:
    """Extract visible SVG labels and accessibility text without visual validation."""
    content = path.read_text(encoding="utf-8")
    if "<!DOCTYPE" in content.upper() or "<!ENTITY" in content.upper():
        msg = "SVG register subjects must not declare a DTD or entities"
        raise ValueError(msg)
    # Authored SVG labels only; DTDs and entity declarations are rejected above.
    root = ET.fromstring(content)  # noqa: S314
    labels = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in {"text", "title", "desc"}:
            labels.append("".join(element.itertext()).strip())
        if "aria-label" in element.attrib:
            labels.append(element.attrib["aria-label"])
    return "\n\n".join(label for label in labels if label) + "\n"


def _subject_content(path: Path, *, commit_message: bool) -> str:
    """Read the prose bytes or SVG labels that both register passes inspect."""
    content = _diagram_labels(path) if path.suffix == ".svg" else path.read_text(encoding="utf-8")
    if commit_message:
        content = (
            "\n".join(line for line in content.splitlines() if not line.startswith("#")) + "\n"
        )
    return content


def run_checks(
    paths: Sequence[Path],
    *,
    reader: Path | None = None,
    tier: str = "internal",
    commit_message: bool = False,
) -> int:
    """Require a successful checker and independent read-through for every subject.

    Args:
        paths: Authored files or a Git commit-message file.
        reader: Optional self-contained reader script, primarily for checker tests.
            The default Codex reader uses rules at ``~/dev/vl-branding``.
        tier: Reader context recorded with the approval.
        commit_message: Whether to omit Git's comment lines from the subject.

    Returns:
        Zero on approval, one on findings, or two on an unavailable checker or bad input.
    """
    if reader is not None and not reader.is_file():
        logger.error("Register reader unavailable: %s. Both passes are required.", reader)
        return 2
    environment = os.environ.copy()
    cache = Path.cwd() / ".register-cache"
    environment.setdefault("REGISTER_STATE_DIR", str(cache))
    environment.setdefault("REGISTER_RECEIPTS", str(cache / "receipts.tsv"))
    cache.mkdir(parents=True, exist_ok=True)
    status = 0
    for path in paths:
        if not path.is_file():
            logger.error("Register subject does not exist: %s", path)
            return 2
        try:
            content = _subject_content(path, commit_message=commit_message)
        except (OSError, UnicodeError, ET.ParseError, ValueError) as error:
            logger.error("Cannot read register subject %s: %s", path, error)
            return 2
        if not content.strip():
            continue
        with TemporaryDirectory(prefix="risk-register-") as directory:
            subject = Path(directory) / "subject.md"
            subject.write_text(content, encoding="utf-8")
            if reader is None:
                outcome = review_with_codex(subject, label=str(path), tier=tier, cache=cache)
            else:
                result = subprocess.run(  # noqa: S603
                    [str(reader), "--tier", tier, "--label", str(path), str(subject)],
                    env=environment,
                    check=False,
                )
                outcome = 1 if result.returncode else 0
            status = max(status, outcome)
            try:
                unchanged = _subject_content(path, commit_message=commit_message) == content
            except (OSError, UnicodeError, ET.ParseError, ValueError) as error:
                logger.error("Cannot recheck register subject %s: %s", path, error)
                return 2
            if not unchanged:
                logger.error("Register subject changed during review: %s. Rerun both passes.", path)
                status = max(status, 1)
    return status


def _authored_paths() -> list[Path]:
    """List tracked and untracked authored prose while excluding generated websites."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],  # noqa: S607
        check=True,
        capture_output=True,
    )
    paths = []
    for name in sorted(set(result.stdout.decode().split("\0")) - {""}):
        path = Path(name)
        if path.name in {"AGENTS.md", "CLAUDE.md"}:
            continue
        if any(name == root or name.startswith(root + "/") for root in _EXCLUDED_ROOTS):
            continue
        if path.suffix in _PROSE_SUFFIXES and path.is_file():
            paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local pre-commit or explicit whole-repository language gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--commit-message", action="store_true")
    parser.add_argument("--tier", choices=("internal", "client"), default="internal")
    arguments = parser.parse_args(argv)
    paths = _authored_paths() if arguments.all else arguments.paths
    if not paths:
        parser.error("provide files or --all")
    return run_checks(
        paths,
        tier=arguments.tier,
        commit_message=arguments.commit_message,
    )


if __name__ == "__main__":
    raise SystemExit(main())
