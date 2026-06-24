"""
Unit tests for the sandboxed exec tool — real subprocess calls (not mocked)
since the behavior under test (cwd confinement, timeout enforcement,
shell=False safety) only shows up when something actually runs.

Cross-platform note: run_command uses shell=False, so it never goes
through cmd.exe or bash — whether a command works depends entirely on
whether that program exists as a standalone executable on the host OS.
`echo`, `ls`, and `sleep` aren't standalone executables on a bare Windows
install (echo is a cmd.exe builtin; ls/sleep are Unix coreutils), so those
specific tests are Linux/macOS-only and skipped on Windows. CI (ubuntu-
latest) always runs the full set. The portable test below uses
sys.executable instead of the bare string "python" so it doesn't depend on
which name happens to be on PATH.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from orchestrator.tools.exec_tools import run_command


@pytest.fixture(autouse=True)
def _sandbox_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("SANDBOX_DIR", str(tmp_path))
    monkeypatch.setenv("TOOL_EXEC_TIMEOUT_SECONDS", "2")
    yield tmp_path


def test_runs_simple_command_and_captures_stdout():
    output = run_command(f'"{sys.executable}" -c "print(\'hello\')"')

    assert "exit_code=0" in output
    assert "hello" in output


@pytest.mark.skipif(sys.platform == "win32", reason="ls is not a standalone executable on Windows")
def test_nonzero_exit_code_is_reported_not_raised():
    output = run_command("ls /no/such/path/at/all")

    assert "exit_code=0" not in output


@pytest.mark.skipif(
    sys.platform == "win32", reason="echo is a cmd.exe builtin, not a standalone executable"
)
def test_shell_metacharacters_are_not_interpreted():
    # With shell=False + shlex.split, ';' is just an argument to `echo`,
    # not a command separator — this is the injection-safety property.
    output = run_command("echo safe;rm -rf /tmp/should-not-run")

    assert "safe;rm" in output


@pytest.mark.skipif(
    sys.platform == "win32", reason="sleep is not a standalone executable on Windows"
)
def test_timeout_is_enforced():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 5")


def test_empty_command_raises():
    with pytest.raises(ValueError):
        run_command("   ")
