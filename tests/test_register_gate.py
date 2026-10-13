"""Checks that manuscript formats reach the real register checker."""

from __future__ import annotations

import subprocess
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


@pytest.mark.parametrize(
    ("verdict", "events", "expected"),
    [
        (
            '{"read_complete":true,"findings":[]}',
            '{"type":"turn.completed","usage":{"output_tokens":100}}',
            0,
        ),
        (
            '{"read_complete":false,"findings":[]}',
            '{"type":"turn.completed","usage":{"output_tokens":100}}',
            2,
        ),
        (
            '{"read_complete":true,"findings":[]}',
            '{"type":"turn.failed","error":{"message":"quota exhausted"}}',
            2,
        ),
        (
            '{"read_complete":true,"findings":[]}',
            '{"type":"item.completed","item":{"type":"command_execution"}}',
            2,
        ),
        ("{}", '{"type":"turn.completed","usage":{"output_tokens":100}}', 2),
        (
            '{"read_complete":true,"findings":[{"quote":"The audit","kind":"metaphor",'
            '"why":"A planted finding.","fix":"Replacement."}]}',
            '{"type":"turn.completed","usage":{"output_tokens":100}}',
            1,
        ),
        (
            '{"read_complete":true,"findings":[{"quote":"Absent text","kind":"metaphor",'
            '"why":"A planted finding.","fix":"Replacement."}]}',
            '{"type":"turn.completed","usage":{"output_tokens":100}}',
            2,
        ),
    ],
)
def test_default_reader_uses_codex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    verdict: str,
    events: str,
    expected: int,
) -> None:
    """Codex is the sole reader; incomplete, malformed, or ungrounded reads cannot approve."""
    home = tmp_path / "home"
    root = home / "dev/vl-branding"
    (root / "checks").mkdir(parents=True)
    (root / "language").mkdir()
    (root / "checks/register-pass.sh").write_text(
        "read -r -d '' BRIEF <<'BRIEF_END'\nRead the whole subject.\nBRIEF_END\n"
    )
    (root / "checks/run-all.sh").write_text("#!/bin/sh\nexit 0\n")
    for name in ("register.md", "ai-tells.md", "sweep.md"):
        (root / "language" / name).write_text("Use plain engineering prose.\n")
    paper = tmp_path / "paper.qmd"
    paper.write_text("The audit records aggregate counts.\n")
    (tmp_path / ".register-model").write_text("gpt-6.1-sol\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/local/bin/codex")
    commands = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Return a complete verdict without contacting an external model."""
        commands.append(command)
        if "--output-last-message" in command:
            answer = Path(command[command.index("--output-last-message") + 1])
            answer.write_text(verdict)
        return subprocess.CompletedProcess(command, 0, stdout=events, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert run_checks([paper]) == expected
    codex_commands = [command for command in commands if Path(command[0]).name == "codex"]
    assert len(codex_commands) == 1
    command = codex_commands[0]
    assert command[command.index("--model") + 1] == "gpt-6.1-sol"
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in command
    assert not any("claude" in str(argument).lower() for argument in command)
    if "quota exhausted" in events:
        assert "quota exhausted" in caplog.text
    if expected == 0:
        assert run_checks([paper]) == 0
        assert sum(Path(cmd[0]).name == "codex" for cmd in commands) == 1
        paper.write_text("The audit records aggregate counts and rates.\n")
        assert run_checks([paper]) == 0
        assert sum(Path(cmd[0]).name == "codex" for cmd in commands) == 2
        (root / "language/register.md").write_text("Keep numeric denominators.\n")
        assert run_checks([paper]) == 0
        assert sum(Path(cmd[0]).name == "codex" for cmd in commands) == 3
        (tmp_path / ".register-model").write_text("different-model\n")
        assert run_checks([paper]) == 0
        assert sum(Path(cmd[0]).name == "codex" for cmd in commands) == 4
        paper.write_text("This subject has no previous approval.\n")
        monkeypatch.setattr("shutil.which", lambda _name: None)
        assert run_checks([paper]) == 2
        assert sum(Path(cmd[0]).name == "codex" for cmd in commands) == 4
