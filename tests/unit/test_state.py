"""
Unit tests for state.py — a real SQLite file in tmp_path, since the point
is verifying what actually persists across separate connections. Each
state.py function opens its own connection, mirroring how a real
restarted process would reopen the db rather than holding one open.
"""

from __future__ import annotations

from orchestrator import state
from orchestrator.config import Config
from orchestrator.orchestrator import Subtask


def _fake_config(tmp_path, **overrides) -> Config:
    defaults = dict(
        llm_provider="gemini",
        gemini_api_key="fake",
        anthropic_api_key="fake",
        db_path=str(tmp_path / "test.db"),
    )
    defaults.update(overrides)
    return Config(**defaults)


def test_save_and_load_plan_round_trip(tmp_path):
    cfg = _fake_config(tmp_path)
    subtasks = [
        Subtask(agent="swe", description="write code"),
        Subtask(agent="swe", description="write tests"),
    ]

    state.save_plan("t1", "build a thing", subtasks, cfg)
    loaded = state.load_subtasks("t1", cfg)

    assert len(loaded) == 2
    assert loaded[0].agent == "swe"
    assert loaded[0].description == "write code"
    assert loaded[0].status == "pending"
    assert loaded[1].description == "write tests"


def test_mark_subtask_running_then_done(tmp_path):
    cfg = _fake_config(tmp_path)
    state.save_plan("t2", "task", [Subtask(agent="swe", description="do it")], cfg)

    state.mark_subtask_running("t2", 0, cfg)
    assert state.load_subtasks("t2", cfg)[0].status == "running"

    state.mark_subtask_done("t2", 0, "all good", cfg)
    record = state.load_subtasks("t2", cfg)[0]
    assert record.status == "done"
    assert "all good" in record.result_json


def test_mark_subtask_failed(tmp_path):
    cfg = _fake_config(tmp_path)
    state.save_plan("t3", "task", [Subtask(agent="swe", description="do it")], cfg)

    state.mark_subtask_failed("t3", 0, "step_limit", cfg)
    record = state.load_subtasks("t3", cfg)[0]
    assert record.status == "failed"
    assert "step_limit" in record.result_json


def test_load_subtasks_for_unknown_task_returns_empty(tmp_path):
    cfg = _fake_config(tmp_path)
    assert state.load_subtasks("does-not-exist", cfg) == []


def test_resuming_partial_plan_preserves_done_status(tmp_path):
    cfg = _fake_config(tmp_path)
    subtasks = [Subtask(agent="swe", description="a"), Subtask(agent="swe", description="b")]
    state.save_plan("t4", "task", subtasks, cfg)
    state.mark_subtask_done("t4", 0, "done a", cfg)

    loaded = state.load_subtasks("t4", cfg)
    assert loaded[0].status == "done"
    assert loaded[1].status == "pending"
