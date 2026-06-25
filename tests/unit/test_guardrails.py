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


# --- role-scoped write restrictions (Phase 4) ----------------------------


def test_tester_can_write_test_files():
    cfg = _fake_config()
    assert (
        check("write_file", {"path": "test_fizzbuzz.py", "content": "x"}, cfg, agent_name="tester")
        is None
    )
    assert (
        check("write_file", {"path": "fizzbuzz_test.py", "content": "x"}, cfg, agent_name="tester")
        is None
    )


def test_tester_cannot_write_implementation_files():
    violation = check(
        "write_file", {"path": "fizzbuzz.py", "content": "x"}, _fake_config(), agent_name="tester"
    )
    assert violation is not None
    assert "test" in violation.reason.lower()


def test_tester_restriction_respects_nested_paths():
    # The check looks at the filename, not just a prefix match on the
    # whole path — "src/tests/test_foo.py" is fine, "src/foo.py" isn't.
    cfg = _fake_config()
    assert (
        check(
            "write_file", {"path": "sub/dir/test_foo.py", "content": "x"}, cfg, agent_name="tester"
        )
        is None
    )
    assert (
        check("write_file", {"path": "sub/dir/foo.py", "content": "x"}, cfg, agent_name="tester")
        is not None
    )


def test_swe_is_not_restricted_to_test_files():
    # The same write that's blocked for "tester" is fine for "swe" (or no
    # agent_name at all) — the restriction is role-scoped, not global.
    cfg = _fake_config()
    assert (
        check("write_file", {"path": "fizzbuzz.py", "content": "x"}, cfg, agent_name="swe") is None
    )
    assert check("write_file", {"path": "fizzbuzz.py", "content": "x"}, cfg) is None


def test_pm_can_write_markdown():
    cfg = _fake_config()
    assert check("write_file", {"path": "BACKLOG.md", "content": "x"}, cfg, agent_name="pm") is None


def test_pm_cannot_write_code():
    violation = check(
        "write_file", {"path": "main.py", "content": "x"}, _fake_config(), agent_name="pm"
    )
    assert violation is not None
    assert "markdown" in violation.reason.lower()


# --- health_check SSRF guard (Phase 4) -----------------------------------


def test_blocks_health_check_to_cloud_metadata_endpoint():
    violation = check(
        "health_check", {"url": "http://169.254.169.254/latest/meta-data/"}, _fake_config()
    )
    assert violation is not None


def test_allows_health_check_to_ordinary_url():
    assert check("health_check", {"url": "https://example.com/health"}, _fake_config()) is None
