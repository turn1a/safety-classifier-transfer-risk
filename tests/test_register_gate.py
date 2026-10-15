"""Checks that manuscript formats reach the real register checker."""

from __future__ import annotations

from pathlib import Path
from shlex import quote

import pytest

from transfer_risk.scripts.register_gate import run_checks


def _checker_only_reader(tmp_path: Path) -> Path:
    """Replace only the external reader while retaining the real keyword checker."""
    root = tmp_path / "branding"
    checker = Path.home() / "dev/vl-branding/checks/check-register.sh"
    if not checker.is_file():
        pytest.skip("local branding language checker is not installed")
    script = root / "checks/register-pass.sh"
    script.parent.mkdir(parents=True)
    script.write_text(f'#!/bin/sh\nfor subject do :; done\nexec "{checker}" "$subject"\n')
    script.chmod(0o755)
    return script


def test_qmd_reaches_the_checker(tmp_path: Path) -> None:
    """A banned phrase in a Quarto paper must fail despite its .qmd extension."""
    paper = tmp_path / "paper.qmd"
    paper.write_text("The system provides a seamless workflow.\n")
    assert run_checks([paper], reader=_checker_only_reader(tmp_path)) == 1


def test_personal_svg_labels_reach_language_checks(tmp_path: Path) -> None:
    """The language gate reads diagram labels without imposing VL visual branding."""
    diagram = tmp_path / "figure.svg"
    diagram.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><text>Seamless evaluation</text></svg>'
    )
    assert run_checks([diagram], reader=_checker_only_reader(tmp_path)) == 1


def test_missing_reader_fails_closed(tmp_path: Path) -> None:
    """An unavailable second-pass reader cannot approve a commit."""
    paper = tmp_path / "paper.qmd"
    paper.write_text("The audit records aggregate counts.\n")
    assert run_checks([paper], reader=tmp_path / "missing-reader") == 2


def test_changes_during_review_cannot_approve_unread_text(tmp_path: Path) -> None:
    """A reader approval covers only the file bytes presented to it."""
    paper = tmp_path / "paper.qmd"
    paper.write_text("The audit records aggregate counts.\n")
    root = tmp_path / "changing-reader"
    reader = root / "checks/register-pass.sh"
    reader.parent.mkdir(parents=True)
    reader.write_text(f"#!/bin/sh\nprintf 'Unread addition.\\n' >> {quote(str(paper))}\nexit 0\n")
    reader.chmod(0o755)
    assert run_checks([paper], reader=reader) == 1


@pytest.mark.parametrize("reader_status", [0, 1, 2])
def test_default_reader_delegates_to_installed_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader_status: int,
) -> None:
    """The installed gate decides approval, including failures and local receipts."""
    home = tmp_path / "home"
    installed = home / "dev/vl-branding/checks/register-pass.sh"
    installed.parent.mkdir(parents=True)
    log = tmp_path / "installed-input.txt"
    installed.write_text(
        "#!/bin/sh\nfor subject do :; done\n"
        f'cat "$subject" > {quote(str(log))}\n'
        'mkdir -p "$REGISTER_STATE_DIR"\n'
        'printf "installed-receipt\\n" >> "$REGISTER_RECEIPTS"\n'
        f"exit {reader_status}\n"
    )
    installed.chmod(0o755)
    paper = tmp_path / "paper.qmd"
    paper.write_text("The audit records aggregate counts.\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    cache = tmp_path / ".register-cache"
    cache.mkdir()
    (cache / "codex-obsolete.json").write_text('{"read_complete": true}')
    assert run_checks([paper]) == reader_status
    assert log.read_text() == paper.read_text()
    assert (cache / "installed/receipts.tsv").read_text() == "installed-receipt\n"


def test_inherited_provider_overrides_cannot_select_another_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provider, model, effort and credentials cannot change the installed reader."""
    home = tmp_path / "home"
    installed = home / "dev/vl-branding/checks/register-pass.sh"
    installed.parent.mkdir(parents=True)
    installed.write_text(
        "#!/bin/sh\n"
        '[ "$REGISTER_PROVIDER" = codex ] || exit 1\n'
        '[ -z "${REGISTER_MODEL+x}${REGISTER_EFFORT+x}" ] || exit 1\n'
        '[ -z "${OPENAI_API_KEY+x}${ANTHROPIC_API_KEY+x}" ] || exit 1\n'
        '[ "$REGISTER_RECEIPTS" = "$PWD/.register-cache/installed/receipts.tsv" ] || exit 1\n'
        "exit 0\n"
    )
    installed.chmod(0o755)
    paper = tmp_path / "paper.txt"
    paper.write_text("The audit records aggregate counts.\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    for name, value in {
        "REGISTER_PROVIDER": "claude",
        "REGISTER_MODEL": "other-model",
        "REGISTER_EFFORT": "low",
        "REGISTER_RECEIPTS": str(tmp_path / "old-receipts.tsv"),
        "OPENAI_API_KEY": "test-placeholder",
        "ANTHROPIC_API_KEY": "test-placeholder",
    }.items():
        monkeypatch.setenv(name, value)
    assert run_checks([paper]) == 0


def test_commit_message_comments_are_excluded(tmp_path: Path) -> None:
    """Git status comments cannot introduce findings into the reviewed message."""
    message = tmp_path / "COMMIT_EDITMSG"
    message.write_text("Validate saved aggregate counts\n# Seamless workflow\n")
    assert run_checks([message], reader=_checker_only_reader(tmp_path), commit_message=True) == 0


def test_svg_entity_declarations_fail_before_reader_execution(tmp_path: Path) -> None:
    """SVG label extraction refuses entity declarations before parsing."""
    diagram = tmp_path / "figure.svg"
    diagram.write_text('<!DOCTYPE svg [<!ENTITY external SYSTEM "file:///etc/passwd">]><svg/>')
    assert run_checks([diagram], reader=_checker_only_reader(tmp_path)) == 2
