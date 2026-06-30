"""
Smoke test for the Streamlit dashboard, using streamlit.testing.v1.AppTest
to actually execute dashboard/app.py headlessly and catch Python-level
exceptions — not just import/syntax checks. This is what caught a real
deprecation warning (use_container_width, removed after 2025-12-31) during
development; a plain syntax check wouldn't have.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_PATH = "dashboard/app.py"


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    """@st.cache_data on dashboard/app.py's loader functions takes no
    arguments, so its cache key doesn't account for which TRACE_DIR/DB_PATH
    env vars were active — harmless in real usage (one long-lived server
    process, a fixed .env), but without this, one test's monkeypatched env
    vars would leak into the next test's AppTest run within the same
    pytest session."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def populated_dirs(tmp_path, monkeypatch):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "abc123.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "type": "llm_call",
                    "step": 1,
                    "latency_seconds": 1.2,
                    "input_tokens": 500,
                    "output_tokens": 50,
                    "estimated_cost_usd": 0.0,
                    "timestamp": 1.0,
                    "task_id": "abc123",
                },
                {
                    "type": "tool_call",
                    "step": 1,
                    "name": "write_file",
                    "arguments": {"path": "x.py"},
                    "output": "Wrote 10 chars",
                    "is_error": False,
                    "blocked": False,
                    "latency_seconds": 0.01,
                    "timestamp": 1.1,
                    "task_id": "abc123",
                },
                {
                    "type": "task_end",
                    "stopped_reason": "done",
                    "steps_taken": 1,
                    "total_input_tokens": 500,
                    "total_output_tokens": 50,
                    "estimated_cost_usd": 0.0,
                    "timestamp": 1.2,
                    "task_id": "abc123",
                },
            ]
        )
        + "\n"
    )

    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        "CREATE TABLE tasks (task_id TEXT PRIMARY KEY, original_task TEXT, "
        "status TEXT, created_at REAL);"
        "CREATE TABLE subtasks (task_id TEXT, subtask_index INTEGER, agent TEXT, description TEXT, "
        "status TEXT, result_json TEXT);"
    )
    conn.execute("INSERT INTO tasks VALUES ('orch1', 'do a thing', 'done', 1.0)")
    conn.execute("INSERT INTO subtasks VALUES ('orch1', 0, 'swe', 'write it', 'done', '\"ok\"')")
    conn.commit()
    conn.close()

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("TRACE_DIR", str(trace_dir))
    monkeypatch.setenv("DB_PATH", str(db_path))
    return trace_dir, db_path


def test_dashboard_runs_without_exceptions_when_data_exists(populated_dirs):
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception


def test_dashboard_shows_metrics_and_task_table(populated_dirs):
    at = AppTest.from_file(APP_PATH)
    at.run()

    labels = [m.label for m in at.metric]
    assert "Tasks run" in labels
    assert "Success rate" in labels
    assert len(at.dataframe) == 1


def test_dashboard_shows_orchestrator_plan(populated_dirs):
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert len(at.expander) == 1
    assert "orch1" in at.expander[0].label


def test_dashboard_handles_no_traces_gracefully(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "empty_traces"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "no_state.db"))

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert any("No traces found" in info.value for info in at.info)
