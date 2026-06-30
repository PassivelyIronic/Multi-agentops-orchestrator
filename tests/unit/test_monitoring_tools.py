"""
Unit tests for monitoring_tools.py: query_traces against real JSONL files
in tmp_path (the same files tracing.py writes), and health_check against a
mocked urlopen so these tests never make a real network call.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from orchestrator.tools import monitoring_tools
from orchestrator.tools.monitoring_tools import health_check, query_traces


@pytest.fixture(autouse=True)
def _trace_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    yield tmp_path


def _write_trace(tmp_path, task_id, records):
    path = tmp_path / f"{task_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# --- query_traces ----------------------------------------------------------


def test_query_traces_with_no_trace_dir_yet(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "does_not_exist"))
    assert "no agent runs" in query_traces().lower()


def test_query_traces_returns_no_match_message_when_empty(tmp_path):
    assert "no matching" in query_traces().lower()


def test_query_traces_filters_by_task_id_prefix(tmp_path):
    _write_trace(tmp_path, "abc-0", [{"type": "tool_call", "is_error": False, "marker": "abc"}])
    _write_trace(tmp_path, "xyz-0", [{"type": "tool_call", "is_error": False, "marker": "xyz"}])

    result = query_traces(task_id_prefix="abc")
    parsed = json.loads(result)

    assert len(parsed) == 1
    assert parsed[0]["marker"] == "abc"


def test_query_traces_filters_by_event_type(tmp_path):
    _write_trace(
        tmp_path,
        "t1",
        [{"type": "llm_call", "input_tokens": 10}, {"type": "tool_call", "is_error": False}],
    )

    result = query_traces(event_type="llm_call")
    parsed = json.loads(result)

    assert len(parsed) == 1
    assert parsed[0]["type"] == "llm_call"


def test_query_traces_errors_only_filters_correctly(tmp_path):
    _write_trace(
        tmp_path,
        "t2",
        [
            {"type": "tool_call", "is_error": False, "blocked": False},
            {"type": "tool_call", "is_error": True, "blocked": False},
            {"type": "tool_call", "is_error": False, "blocked": True},
            {"type": "llm_error", "error": "boom"},
        ],
    )

    result = query_traces(errors_only=True)
    parsed = json.loads(result)

    assert len(parsed) == 3  # the one clean tool_call is excluded


def test_query_traces_skips_malformed_lines(tmp_path):
    path = tmp_path / "t3.jsonl"
    path.write_text("not json\n" + json.dumps({"type": "tool_call", "is_error": False}) + "\n")

    result = query_traces()

    assert json.loads(result) == [{"type": "tool_call", "is_error": False}]


# --- health_check ------------------------------------------------------------


def test_health_check_rejects_non_http_scheme():
    with pytest.raises(ValueError):
        health_check("ftp://example.com")


def test_health_check_reports_status_on_success(monkeypatch):
    class _FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(monitoring_tools.urllib.request, "urlopen", lambda *a, **k: _FakeResponse())

    result = health_check("https://example.com")

    assert "status=200" in result


def test_health_check_reports_http_error_without_raising(monkeypatch):
    def _raise_http_error(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.com", 503, "Service Unavailable", None, None)

    monkeypatch.setattr(monitoring_tools.urllib.request, "urlopen", _raise_http_error)

    result = health_check("https://example.com")

    assert "status=503" in result


def test_health_check_reports_connection_failure_without_raising(monkeypatch):
    def _raise_generic(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(monitoring_tools.urllib.request, "urlopen", _raise_generic)

    result = health_check("https://example.com")

    assert "could not connect" in result
