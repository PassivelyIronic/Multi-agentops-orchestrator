"""
Unit tests for orchestrator.py: decomposition parsing (no LLM involved),
and run()'s routing/checkpointing/resume logic (LLM call and agent classes
both faked — no real API calls or BaseAgent loop here, that's already
covered by test_base_agent.py).
"""

from __future__ import annotations

import json

import pytest

from orchestrator import orchestrator as orchestrator_module
from orchestrator import state
from orchestrator.agents.base_agent import AgentResult
from orchestrator.config import Config
from orchestrator.llm_client import LLMResponse
from orchestrator.orchestrator import (
    AGENT_DESCRIPTIONS,
    AVAILABLE_AGENTS,
    Subtask,
    _parse_subtasks,
    run,
)


def _fake_config(tmp_path, **overrides) -> Config:
    defaults = dict(
        llm_provider="gemini",
        gemini_api_key="fake",
        anthropic_api_key="fake",
        db_path=str(tmp_path / "test.db"),
    )
    defaults.update(overrides)
    return Config(**defaults)


class _FakeAgent:
    """Stands in for a real agent class — same .run() interface, no LLM calls."""

    calls: list[str] = []

    def __init__(self, config=None):
        self.config = config

    def run(self, description, task_id=None):
        _FakeAgent.calls.append(description)
        return AgentResult(final_text=f"did: {description}", steps_taken=1, stopped_reason="done")


class _FailingAgent:
    def __init__(self, config=None):
        self.config = config

    def run(self, description, task_id=None):
        return AgentResult(final_text=None, steps_taken=1, stopped_reason="step_limit")


@pytest.fixture(autouse=True)
def _reset_fake_agent_calls():
    _FakeAgent.calls = []
    yield


# --- decomposition parsing ---------------------------------------------


def test_parse_subtasks_valid_json():
    raw = json.dumps([{"agent": "swe", "description": "write code"}])
    assert _parse_subtasks(raw, ["swe"], fallback_task="fallback") == [
        Subtask(agent="swe", description="write code")
    ]


def test_parse_subtasks_strips_markdown_fences():
    raw = "```json\n" + json.dumps([{"agent": "swe", "description": "x"}]) + "\n```"
    assert _parse_subtasks(raw, ["swe"], fallback_task="fallback") == [
        Subtask(agent="swe", description="x")
    ]


def test_parse_subtasks_falls_back_on_malformed_json():
    result = _parse_subtasks("not json at all", ["swe"], fallback_task="fallback task")
    assert result == [Subtask(agent="swe", description="fallback task")]


def test_parse_subtasks_falls_back_on_unknown_agent():
    raw = json.dumps([{"agent": "does_not_exist", "description": "x"}])
    result = _parse_subtasks(raw, ["swe"], fallback_task="fallback task")
    assert result == [Subtask(agent="swe", description="fallback task")]


def test_parse_subtasks_falls_back_on_empty_text():
    result = _parse_subtasks(None, ["swe"], fallback_task="fallback task")
    assert result == [Subtask(agent="swe", description="fallback task")]


# --- run(): routing, checkpointing, resume -----------------------------


def test_run_decomposes_and_routes_single_subtask(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator_module, "AVAILABLE_AGENTS", {"swe": _FakeAgent})
    monkeypatch.setattr(
        orchestrator_module,
        "call_with_tools",
        lambda *a, **k: LLMResponse(
            text=json.dumps([{"agent": "swe", "description": "write hello.txt"}])
        ),
    )

    result = run("create a file", config=_fake_config(tmp_path), task_id="task-1")

    assert result.stopped_reason == "done"
    assert len(result.subtask_results) == 1
    assert _FakeAgent.calls == ["write hello.txt"]


def test_run_routes_multiple_subtasks_in_order(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator_module, "AVAILABLE_AGENTS", {"swe": _FakeAgent})
    plan = [
        {"agent": "swe", "description": "step one"},
        {"agent": "swe", "description": "step two"},
    ]
    monkeypatch.setattr(
        orchestrator_module, "call_with_tools", lambda *a, **k: LLMResponse(text=json.dumps(plan))
    )

    result = run("multi-step task", config=_fake_config(tmp_path), task_id="task-2")

    assert result.stopped_reason == "done"
    assert _FakeAgent.calls == ["step one", "step two"]


def test_run_stops_on_subtask_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator_module, "AVAILABLE_AGENTS", {"swe": _FailingAgent})
    monkeypatch.setattr(
        orchestrator_module,
        "call_with_tools",
        lambda *a, **k: LLMResponse(text=json.dumps([{"agent": "swe", "description": "x"}])),
    )

    result = run("a task", config=_fake_config(tmp_path), task_id="task-3")

    assert result.stopped_reason == "subtask_failed"


def test_run_propagates_subtask_error_message(monkeypatch, tmp_path):
    """An api_error from a subtask (e.g. exhausted retries on a 429) should
    surface its .error message on the OrchestratorResult too, not just the
    generic 'subtask_failed' reason."""

    class _ApiErrorAgent:
        def __init__(self, config=None):
            self.config = config

        def run(self, description, task_id=None):
            return AgentResult(
                final_text=None,
                steps_taken=0,
                stopped_reason="api_error",
                error="429 RESOURCE_EXHAUSTED: daily quota exceeded",
            )

    monkeypatch.setattr(orchestrator_module, "AVAILABLE_AGENTS", {"swe": _ApiErrorAgent})
    monkeypatch.setattr(
        orchestrator_module,
        "call_with_tools",
        lambda *a, **k: LLMResponse(text=json.dumps([{"agent": "swe", "description": "x"}])),
    )

    result = run("a task", config=_fake_config(tmp_path), task_id="task-3b")

    assert result.stopped_reason == "subtask_failed"
    assert result.error is not None
    assert "429" in result.error


def test_run_handles_decomposition_failure_gracefully(monkeypatch, tmp_path):
    """If the decomposition LLM call itself exhausts retries (e.g. daily
    quota gone before any plan exists), run() must return a clean result —
    not crash the whole script — and must not have saved a partial plan,
    since a retry with the same task_id should re-decompose from scratch."""
    monkeypatch.setattr(
        orchestrator_module,
        "call_with_tools",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("429 RESOURCE_EXHAUSTED")),
    )

    cfg = _fake_config(tmp_path)
    result = run("a task", config=cfg, task_id="task-6")

    assert result.stopped_reason == "decomposition_error"
    assert result.error is not None
    assert "429" in result.error
    assert state.load_subtasks("task-6", cfg) == []  # nothing partially saved


def test_run_rejects_unknown_agent_in_plan(tmp_path):
    # Decomposition itself only ever returns valid agents (_parse_subtasks
    # guarantees that), so to exercise this path we pre-seed state with a
    # plan referencing an agent no longer in AVAILABLE_AGENTS — simulating
    # a config change between a crash and a resume.
    cfg = _fake_config(tmp_path)
    state.save_plan("task-4", "a task", [Subtask(agent="retired_agent", description="x")], cfg)

    result = run("a task", config=cfg, task_id="task-4")

    assert result.stopped_reason == "unknown_agent"


def test_run_resumes_after_partial_completion(monkeypatch, tmp_path):
    """Simulates a crash after subtask 1: state has it marked done, and a
    fresh run() call with the same task_id must skip it, never re-decompose,
    and only run the remaining subtask."""
    cfg = _fake_config(tmp_path)
    plan = [
        Subtask(agent="swe", description="already done"),
        Subtask(agent="swe", description="still pending"),
    ]
    state.save_plan("task-5", "a task", plan, cfg)
    state.mark_subtask_done("task-5", 0, "finished earlier", cfg)

    monkeypatch.setattr(orchestrator_module, "AVAILABLE_AGENTS", {"swe": _FakeAgent})
    decompose_called = {"hit": False}
    monkeypatch.setattr(
        orchestrator_module,
        "call_with_tools",
        lambda *a, **k: decompose_called.update(hit=True),
    )

    result = run("a task", config=cfg, task_id="task-5")

    assert decompose_called["hit"] is False  # resumed from saved plan, never re-decomposed
    assert _FakeAgent.calls == ["still pending"]
    assert result.stopped_reason == "done"


# --- Phase 4: all four agent roles wired consistently --------------------


def test_all_four_agent_roles_are_registered():
    assert set(AVAILABLE_AGENTS.keys()) == {"swe", "tester", "oncall", "pm"}


def test_every_available_agent_has_a_decomposition_description():
    # AGENT_DESCRIPTIONS feeds the decomposition prompt — if an agent is
    # routable but undescribed, the model has no way to know when to pick it.
    for name in AVAILABLE_AGENTS:
        assert name in AGENT_DESCRIPTIONS
        assert AGENT_DESCRIPTIONS[name].strip()


def test_agent_classes_self_identify_with_matching_key():
    # AVAILABLE_AGENTS is built from each class's own agent_name — this
    # guards against the dict key and the class's self-identification
    # (used by guardrails.py's role checks) ever drifting apart.
    for name, cls in AVAILABLE_AGENTS.items():
        assert cls.agent_name == name
