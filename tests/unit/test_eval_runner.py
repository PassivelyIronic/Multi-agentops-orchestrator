"""
Unit tests for eval_runner.py: golden dataset loading, each verification
type, and run_task's routing/isolation — agents and the orchestrator are
faked (same pattern as test_orchestrator.py's _FakeAgent), so these never
make a real API call.
"""

from __future__ import annotations

import json

from orchestrator import eval_runner as eval_runner_module
from orchestrator.agents.base_agent import AgentResult
from orchestrator.config import Config
from orchestrator.eval_runner import (
    EvalReport,
    GoldenTask,
    TaskEvalResult,
    _verify_all,
    _verify_one,
    load_golden_dataset,
    run_task,
)


def _fake_config(**overrides) -> Config:
    defaults = dict(llm_provider="gemini", gemini_api_key="fake", anthropic_api_key="fake")
    defaults.update(overrides)
    return Config(**defaults)


# --- load_golden_dataset ---------------------------------------------------


def test_load_golden_dataset_round_trip(tmp_path):
    path = tmp_path / "ds.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "t1",
                "agent": "swe",
                "task": "do something",
                "verification": {"type": "file_exists", "path": "x.py"},
                "setup_sandbox": {"a.txt": "hi"},
            }
        )
        + "\n"
    )

    tasks = load_golden_dataset(path)

    assert len(tasks) == 1
    assert tasks[0].id == "t1"
    assert tasks[0].agent == "swe"
    assert tasks[0].setup_sandbox == {"a.txt": "hi"}


def test_load_golden_dataset_skips_blank_lines(tmp_path):
    path = tmp_path / "ds.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "t1",
                "agent": "swe",
                "task": "x",
                "verification": {"type": "file_exists", "path": "a"},
            }
        )
        + "\n\n\n"
    )

    assert len(load_golden_dataset(path)) == 1


# --- verification types -----------------------------------------------------


def test_verify_file_exists(tmp_path):
    (tmp_path / "a.py").write_text("x")
    assert (
        _verify_one({"type": "file_exists", "path": "a.py"}, tmp_path, "", _fake_config())[0]
        is True
    )
    assert (
        _verify_one({"type": "file_exists", "path": "b.py"}, tmp_path, "", _fake_config())[0]
        is False
    )


def test_verify_file_contains(tmp_path):
    (tmp_path / "a.py").write_text("def greet(): pass")
    spec = {"type": "file_contains", "path": "a.py", "substring": "greet"}
    assert _verify_one(spec, tmp_path, "", _fake_config())[0] is True


def test_verify_file_unchanged(tmp_path):
    (tmp_path / "a.py").write_text("original")
    matches_spec = {"type": "file_unchanged", "path": "a.py", "expected_content": "original"}
    differs_spec = {"type": "file_unchanged", "path": "a.py", "expected_content": "different"}
    assert _verify_one(matches_spec, tmp_path, "", _fake_config())[0] is True
    assert _verify_one(differs_spec, tmp_path, "", _fake_config())[0] is False


def test_verify_no_files_matching(tmp_path):
    spec = {"type": "no_files_matching", "pattern": "*.py"}
    assert _verify_one(spec, tmp_path, "", _fake_config())[0] is True

    (tmp_path / "oops.py").write_text("x")
    assert _verify_one(spec, tmp_path, "", _fake_config())[0] is False


def test_verify_command_succeeds(tmp_path):
    spec = {"type": "command_succeeds", "command": "python -c \"print('ok')\""}
    passed, detail = _verify_one(spec, tmp_path, "", _fake_config())
    assert passed is True


def test_verify_command_fails_reports_exit_code(tmp_path):
    spec = {"type": "command_succeeds", "command": 'python -c "import sys; sys.exit(1)"'}
    passed, detail = _verify_one(spec, tmp_path, "", _fake_config())
    assert passed is False
    assert "exit_code=1" in detail


def test_verify_llm_judge_uses_final_text_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(
        eval_runner_module, "judge", lambda rubric, context, cfg: (context == "the text", "r")
    )
    spec = {"type": "llm_judge", "rubric": "whatever"}

    passed, _ = _verify_one(spec, tmp_path, "the text", _fake_config())

    assert passed is True


def test_verify_llm_judge_uses_context_file_when_given(monkeypatch, tmp_path):
    (tmp_path / "BACKLOG.md").write_text("file content")
    monkeypatch.setattr(
        eval_runner_module, "judge", lambda rubric, context, cfg: (context == "file content", "r")
    )
    spec = {"type": "llm_judge", "rubric": "whatever", "context_file": "BACKLOG.md"}

    passed, _ = _verify_one(spec, tmp_path, "ignored final_text", _fake_config())

    assert passed is True


def test_verify_unknown_type_fails():
    passed, detail = _verify_one({"type": "nonsense"}, None, "", _fake_config())
    assert passed is False


def test_verify_all_requires_every_spec_to_pass(tmp_path):
    (tmp_path / "a.py").write_text("x")
    specs = [{"type": "file_exists", "path": "a.py"}, {"type": "file_exists", "path": "missing.py"}]

    passed, detail = _verify_all(specs, tmp_path, "", _fake_config())

    assert passed is False
    assert "a.py" in detail and "missing.py" in detail


# --- run_task: routing + isolation ------------------------------------------


class _FakeAgent:
    last_config = None

    def __init__(self, config=None):
        self.config = config
        _FakeAgent.last_config = config

    def run(self, description, task_id=None):
        # Write into whatever sandbox_dir it was actually given — proves
        # isolation (the test checks this landed under work_root, not the
        # shared ./workspace).
        from pathlib import Path

        Path(self.config.sandbox_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.config.sandbox_dir) / "output.txt").write_text("done")
        return AgentResult(final_text="all good", steps_taken=1, stopped_reason="done", error=None)


def test_run_task_isolates_sandbox_per_task(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_runner_module, "AVAILABLE_AGENTS", {"swe": _FakeAgent})
    task = GoldenTask(
        id="iso_test",
        agent="swe",
        task="do it",
        verification={"type": "file_exists", "path": "output.txt"},
    )

    result = run_task(task, work_root=tmp_path, base_config=_fake_config())

    assert result.passed is True
    assert str(tmp_path) in _FakeAgent.last_config.sandbox_dir
    assert (tmp_path / "iso_test" / "workspace" / "output.txt").is_file()


def test_run_task_seeds_sandbox_files_before_running(monkeypatch, tmp_path):
    seen = {}

    class _CheckingAgent:
        def __init__(self, config=None):
            self.config = config

        def run(self, description, task_id=None):
            from pathlib import Path

            seen["seeded_content"] = (Path(self.config.sandbox_dir) / "existing.py").read_text()
            return AgentResult(final_text="ok", steps_taken=1, stopped_reason="done")

    monkeypatch.setattr(eval_runner_module, "AVAILABLE_AGENTS", {"swe": _CheckingAgent})
    task = GoldenTask(
        id="seed_test",
        agent="swe",
        task="do it",
        verification={"type": "file_exists", "path": "existing.py"},
        setup_sandbox={"existing.py": "print('seeded')"},
    )

    run_task(task, work_root=tmp_path, base_config=_fake_config())

    assert seen["seeded_content"] == "print('seeded')"


def test_run_task_handles_unknown_agent_name(tmp_path):
    task = GoldenTask(
        id="bad_agent",
        agent="not_a_real_agent",
        task="x",
        verification={"type": "file_exists", "path": "x"},
    )

    result = run_task(task, work_root=tmp_path, base_config=_fake_config())

    assert result.passed is False
    assert result.error is not None


def test_run_task_catches_agent_exceptions_as_a_safety_net(monkeypatch, tmp_path):
    class _ExplodingAgent:
        def __init__(self, config=None):
            pass

        def run(self, description, task_id=None):
            raise RuntimeError("unexpected crash")

    monkeypatch.setattr(eval_runner_module, "AVAILABLE_AGENTS", {"swe": _ExplodingAgent})
    task = GoldenTask(
        id="boom", agent="swe", task="x", verification={"type": "file_exists", "path": "x"}
    )

    result = run_task(task, work_root=tmp_path, base_config=_fake_config())

    assert result.passed is False
    assert "unexpected crash" in result.error


def test_run_task_routes_to_orchestrator(monkeypatch, tmp_path):
    from orchestrator.orchestrator import OrchestratorResult, Subtask

    fake_result = OrchestratorResult(
        task_id="x",
        subtask_results=[(Subtask(agent="swe", description="d"), AgentResult("final", 1, "done"))],
        stopped_reason="done",
    )
    monkeypatch.setattr(
        eval_runner_module, "run_orchestrator", lambda task, config=None: fake_result
    )

    task = GoldenTask(
        id="orch_test",
        agent="orchestrator",
        task="x",
        verification={"type": "file_exists", "path": "nope"},
    )
    result = run_task(task, work_root=tmp_path, base_config=_fake_config())

    assert result.stopped_reason == "done"


# --- EvalReport --------------------------------------------------------------


def test_eval_report_pass_rate_and_token_totals():
    results = [
        TaskEvalResult("t1", True, "ok", "done", 2, 100, 50, 1.0),
        TaskEvalResult("t2", False, "failed", "step_limit", 5, 200, 80, 2.0),
    ]
    report = EvalReport(results)

    assert report.pass_count == 1
    assert report.pass_rate == 0.5
    assert report.total_input_tokens == 300
    assert report.total_output_tokens == 130


def test_eval_report_markdown_includes_failure_details():
    results = [
        TaskEvalResult("t1", False, "verification detail here", "done", 1, 10, 5, 0.5, error="boom")
    ]
    md = EvalReport(results).to_markdown()

    assert "t1" in md
    assert "FAIL" in md
    assert "verification detail here" in md
    assert "boom" in md


def test_run_task_with_real_agent_and_real_tools_isolates_correctly(monkeypatch, tmp_path):
    """End-to-end regression for the exact bug caught live in Phase 5: the
    existing isolation tests above use a fake agent that manually writes
    into self.config.sandbox_dir, bypassing the real tool-calling path
    entirely — they'd have passed even with the bug, since the bug lived
    specifically in the seam between an agent's Config and the real tool
    functions it calls. This uses the real SweAgent + real filesystem_tools
    through run_task itself, with only the LLM call mocked, to prove the
    full eval pipeline actually isolates end-to-end."""
    import orchestrator.agents.base_agent as base_agent_module
    from orchestrator.agents.swe_agent import SweAgent
    from orchestrator.llm_client import LLMResponse, ToolCall
    from orchestrator.tools import filesystem_tools  # noqa: F401  (registers real tools)

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    # The "wrong" sandbox a buggy tool would fall back to — deliberately
    # different from where run_task isolates this task's real sandbox.
    monkeypatch.setenv("SANDBOX_DIR", str(tmp_path / "wrong_global_sandbox"))

    monkeypatch.setattr(eval_runner_module, "AVAILABLE_AGENTS", {"swe": SweAgent})

    responses = [
        LLMResponse(
            text=None,
            tool_calls=[
                ToolCall(id="1", name="write_file", arguments={"path": "out.txt", "content": "hi"})
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    monkeypatch.setattr(base_agent_module, "call_with_tools", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(
        base_agent_module, "to_assistant_turn", lambda *a, **k: [{"role": "assistant"}]
    )
    monkeypatch.setattr(
        base_agent_module, "to_tool_result_turn", lambda *a, **k: [{"role": "user"}]
    )

    task = GoldenTask(
        id="real_isolation_test",
        agent="swe",
        task="write a file",
        verification={"type": "file_exists", "path": "out.txt"},
    )

    result = run_task(task, work_root=tmp_path / "runs")

    assert result.passed is True
    assert (tmp_path / "runs" / "real_isolation_test" / "workspace" / "out.txt").is_file()
    assert not (tmp_path / "wrong_global_sandbox" / "out.txt").exists()
