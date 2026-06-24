"""
Unit tests for guardrails.check() — the policy layer consulted before a
tool runs. These call check() directly with plain dicts, no agent loop or
registry involved, since the only thing under test is the matching logic.
"""

from __future__ import annotations

from orchestrator.config import Config
from orchestrator.guardrails import check


def _fake_config(**overrides) -> Config:
    defaults = dict(llm_provider="gemini", gemini_api_key="fake", anthropic_api_key="fake")
    defaults.update(overrides)
    return Config(**defaults)


def test_allows_safe_command():
    assert check("run_command", {"command": "pytest -v"}, _fake_config()) is None


def test_allows_git_status_and_diff():
    assert check("run_command", {"command": "git status"}, _fake_config()) is None
    assert check("run_command", {"command": "git diff"}, _fake_config()) is None


def test_blocks_rm():
    violation = check("run_command", {"command": "rm -rf ./workspace"}, _fake_config())
    assert violation is not None
    assert "blocked" in violation.reason.lower()


def test_blocks_git_push():
    violation = check("run_command", {"command": "git push origin main"}, _fake_config())
    assert violation is not None


def test_blocks_pip_install():
    violation = check("run_command", {"command": "pip install requests"}, _fake_config())
    assert violation is not None


def test_blocks_sudo():
    violation = check("run_command", {"command": "sudo apt-get update"}, _fake_config())
    assert violation is not None


def test_custom_denied_patterns_override_defaults():
    cfg = _fake_config(denied_command_patterns=(r"\bpytest\b",))
    # "rm" is no longer denied once custom patterns are supplied — they replace,
    # not extend, the default list.
    assert check("run_command", {"command": "rm -rf /"}, cfg) is None
    assert check("run_command", {"command": "pytest"}, cfg) is not None


def test_allows_small_write():
    violation = check("write_file", {"path": "a.txt", "content": "hello"}, _fake_config())
    assert violation is None


def test_blocks_write_exceeding_max_bytes():
    cfg = _fake_config(max_write_bytes=10)
    violation = check(
        "write_file", {"path": "a.txt", "content": "this is way more than 10 bytes"}, cfg
    )
    assert violation is not None


def test_unrelated_tool_is_never_blocked():
    assert check("read_file", {"path": "anything"}, _fake_config()) is None
