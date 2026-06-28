"""
Unit tests for BaseAgent's loop: step limits, task timeout, token budget,
repetition detection, tool-error isolation, tool-scope enforcement,
guardrail blocking, and trace logging — all without any real API calls.
call_with_tools / to_assistant_turn / to_tool_result_turn are monkeypatched
at the module level and fed a scripted sequence of LLMResponse objects, so
these tests run instantly and need no API key.

Every test routes trace_dir to tmp_path — run() always creates a real
TraceLogger, so without this every test run would write JSONL files into
a real ./traces directory on disk instead of a throwaway temp one.
"""

from __future__ import annotations

import json

from orchestrator.agents import base_agent as base_agent_module
from orchestrator.agents.base_agent import BaseAgent
from orchestrator.config import Config
from orchestrator.llm_client import LLMResponse, ToolCall
from orchestrator.tools.registry import ToolRegistry, ToolSpec
from orchestrator.tracing import TraceLogger


def _fake_config(trace_dir, **overrides) -> Config:
    defaults = dict(
        llm_provider="gemini",
        gemini_api_key="fake",
        anthropic_api_key="fake",
        trace_dir=str(trace_dir),
    )
    defaults.update(overrides)
    return Config(**defaults)


def _registry_with_echo_tool() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="echo",
            description="Echoes back its input.",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            fn=lambda text: f"echo: {text}",
        )
    )
    return reg


def _stub_turn_builders(monkeypatch):
    """The loop only cares that *something* gets appended to history between
    steps — the exact shape is llm_client's concern and already covered by
    its own tests, so we stub it out here to keep these tests focused.
    Both real functions return lists now (base_agent.py uses .extend()),
    so the stubs must too."""
    monkeypatch.setattr(
        base_agent_module, "to_assistant_turn", lambda *a, **k: [{"role": "assistant"}]
    )
    monkeypatch.setattr(
        base_agent_module, "to_tool_result_turn", lambda *a, **k: [{"role": "user"}]
    )


class _EchoAgent(BaseAgent):
    system_prompt = "test agent"
    tool_names = ["echo"]


def test_stops_when_no_tool_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(
        base_agent_module,
        "call_with_tools",
        lambda *a, **k: LLMResponse(text="all done", tool_calls=[]),
    )
    agent = _EchoAgent(config=_fake_config(tmp_path), tool_registry=_registry_with_echo_tool())

    result = agent.run("do the thing")

    assert result.stopped_reason == "done"
    assert result.final_text == "all done"
    assert result.steps_taken == 1


def test_executes_tool_then_stops(monkeypatch, tmp_path):
    responses = [
        LLMResponse(
            text=None, tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "hi"})]
        ),
        LLMResponse(text="finished", tool_calls=[]),
    ]
    monkeypatch.setattr(base_agent_module, "call_with_tools", lambda *a, **k: responses.pop(0))
    _stub_turn_builders(monkeypatch)

    agent = _EchoAgent(config=_fake_config(tmp_path), tool_registry=_registry_with_echo_tool())
    result = agent.run("do the thing")

    assert result.stopped_reason == "done"
    assert result.final_text == "finished"
    assert result.steps_taken == 2


def test_hits_step_limit(monkeypatch, tmp_path):
    always_calls_tool = LLMResponse(
        text=None, tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "x"})]
    )
    monkeypatch.setattr(base_agent_module, "call_with_tools", lambda *a, **k: always_calls_tool)
    _stub_turn_builders(monkeypatch)

    # repetition_limit set high so the step limit is what triggers, not the repetition guard
    agent = _EchoAgent(
        config=_fake_config(tmp_path, max_steps_per_agent=3, repetition_limit=100),
        tool_registry=_registry_with_echo_tool(),
    )
    result = agent.run("do the thing")

    assert result.stopped_reason == "step_limit"
    assert result.steps_taken == 3


def test_detects_repetition(monkeypatch, tmp_path):
    same_call_every_time = LLMResponse(
        text=None, tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "x"})]
    )
    monkeypatch.setattr(base_agent_module, "call_with_tools", lambda *a, **k: same_call_every_time)
    _stub_turn_builders(monkeypatch)

    agent = _EchoAgent(
        config=_fake_config(tmp_path, max_steps_per_agent=20, repetition_limit=3),
        tool_registry=_registry_with_echo_tool(),
    )
    result = agent.run("do the thing")

    assert result.stopped_reason == "repetition"
    assert result.steps_taken < 20


def test_token_budget_is_enforced(monkeypatch, tmp_path):
    monkeypatch.setattr(
        base_agent_module,
        "call_with_tools",
        lambda *a, **k: LLMResponse(
            text="huge response", tool_calls=[], input_tokens=600_000, output_tokens=0
        ),
    )

    agent = _EchoAgent(
        config=_fake_config(tmp_path, max_tokens_per_task=100_000),
        tool_registry=_registry_with_echo_tool(),
    )
    result = agent.run("do the thing")

    assert result.stopped_reason == "token_budget"


def test_isolates_tool_errors(monkeypatch, tmp_path):
    """A tool that raises should not crash the loop — it becomes an error
    result fed back to the model, which then gets a chance to recover."""
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="boom",
            description="Always fails.",
            input_schema={"type": "object", "properties": {}},
            fn=lambda: (_ for _ in ()).throw(RuntimeError("kaboom")),
        )
    )

    responses = [
        LLMResponse(text=None, tool_calls=[ToolCall(id="1", name="boom", arguments={})]),
        LLMResponse(text="recovered", tool_calls=[]),
    ]
    monkeypatch.setattr(base_agent_module, "call_with_tools", lambda *a, **k: responses.pop(0))
    _stub_turn_builders(monkeypatch)

    class _BoomAgent(BaseAgent):
        system_prompt = "test"
        tool_names = ["boom"]

    agent = _BoomAgent(config=_fake_config(tmp_path), tool_registry=reg)
    result = agent.run("trigger the bug")

    assert result.stopped_reason == "done"
    assert result.final_text == "recovered"


def test_task_timeout_is_enforced(monkeypatch, tmp_path):
    import time

    def slow_call(*args, **kwargs):
        time.sleep(0.05)
        return LLMResponse(
            text=None, tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "x"})]
        )

    monkeypatch.setattr(base_agent_module, "call_with_tools", slow_call)
    _stub_turn_builders(monkeypatch)

    agent = _EchoAgent(
        config=_fake_config(
            tmp_path, task_timeout_seconds=0, max_steps_per_agent=1000, repetition_limit=1000
        ),
        tool_registry=_registry_with_echo_tool(),
    )
    result = agent.run("do the thing")

    assert result.stopped_reason == "task_timeout"


def test_tool_outside_agent_scope_is_rejected(monkeypatch, tmp_path):
    """A tool that exists in the shared registry but isn't in *this* agent's
    tool_names must never run — even if the model hallucinates a call to it."""
    called = {"hit": False}

    def secret_fn():
        called["hit"] = True
        return "should never run"

    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="echo",
            description="Echoes back its input.",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            fn=lambda text: f"echo: {text}",
        )
    )
    reg.register(
        ToolSpec(
            name="secret_tool",
            description="Not granted to this agent.",
            input_schema={"type": "object", "properties": {}},
            fn=secret_fn,
        )
    )

    responses = [
        LLMResponse(text=None, tool_calls=[ToolCall(id="1", name="secret_tool", arguments={})]),
        LLMResponse(text="done", tool_calls=[]),
    ]
    monkeypatch.setattr(base_agent_module, "call_with_tools", lambda *a, **k: responses.pop(0))
    _stub_turn_builders(monkeypatch)

    agent = _EchoAgent(
        config=_fake_config(tmp_path), tool_registry=reg
    )  # tool_names = ["echo"] only
    result = agent.run("try the secret tool")

    assert called["hit"] is False
    assert result.stopped_reason == "done"


def test_guardrail_blocks_dangerous_command_before_execution(monkeypatch, tmp_path):
    """rm -rf should never reach the real tool function — guardrails.check
    intercepts it first, using the actual (non-mocked) guardrail logic."""
    called = {"hit": False}

    def fake_run_command(command):
        called["hit"] = True
        return "should never run"

    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="run_command",
            description="Runs a command.",
            input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
            fn=fake_run_command,
        )
    )

    responses = [
        LLMResponse(
            text=None,
            tool_calls=[ToolCall(id="1", name="run_command", arguments={"command": "rm -rf /"})],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    monkeypatch.setattr(base_agent_module, "call_with_tools", lambda *a, **k: responses.pop(0))
    _stub_turn_builders(monkeypatch)

    class _RunnerAgent(BaseAgent):
        system_prompt = "test"
        tool_names = ["run_command"]

    agent = _RunnerAgent(config=_fake_config(tmp_path), tool_registry=reg)
    result = agent.run("do something dangerous")

    assert called["hit"] is False
    assert result.stopped_reason == "done"


def test_llm_call_failure_is_caught_gracefully(monkeypatch, tmp_path):
    """An exception escaping call_with_tools (e.g. retries exhausted on a
    429 / exhausted daily quota) must not crash the whole task — it
    becomes a graceful AgentResult, the same way a bad tool call already
    does. This is what lets orchestrator.py mark the subtask failed and
    resume it later instead of the whole process dying."""

    def always_raises(*args, **kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: daily quota exceeded")

    monkeypatch.setattr(base_agent_module, "call_with_tools", always_raises)

    agent = _EchoAgent(config=_fake_config(tmp_path), tool_registry=_registry_with_echo_tool())
    result = agent.run("do the thing")

    assert result.stopped_reason == "api_error"
    assert result.error is not None
    assert "429" in result.error
    assert result.steps_taken == 0


def test_llm_call_failure_is_logged_to_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(
        base_agent_module,
        "call_with_tools",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    agent = _EchoAgent(config=_fake_config(tmp_path), tool_registry=_registry_with_echo_tool())
    result = agent.run("do the thing")

    trace_path = tmp_path / f"{result.task_id}.jsonl"
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    types_seen = {r["type"] for r in records}
    assert "llm_error" in types_seen
    assert "task_end" in types_seen


def test_trace_file_is_written(monkeypatch, tmp_path):
    monkeypatch.setattr(
        base_agent_module,
        "call_with_tools",
        lambda *a, **k: LLMResponse(text="done", tool_calls=[]),
    )

    agent = _EchoAgent(config=_fake_config(tmp_path), tool_registry=_registry_with_echo_tool())
    result = agent.run("do the thing")

    trace_path = tmp_path / f"{result.task_id}.jsonl"
    assert trace_path.exists()

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    types_seen = {r["type"] for r in records}
    assert "llm_call" in types_seen
    assert "task_end" in types_seen


def test_role_scoped_guardrail_blocks_before_real_tool_runs(monkeypatch, tmp_path):
    """End-to-end wiring check (Phase 4): an agent identifying as "tester"
    must have its write_file call to a non-test path blocked by the real
    (non-mocked) guardrails.check — and the real write_file function must
    never actually execute. This is the integration point between
    base_agent.py passing self.agent_name through and guardrails.py's
    role-scoped rule actually using it."""
    written = {"hit": False}

    def fake_write_file(path, content):
        written["hit"] = True
        return f"wrote {path}"

    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="write_file",
            description="Writes a file.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            },
            fn=fake_write_file,
        )
    )

    responses = [
        LLMResponse(
            text=None,
            tool_calls=[
                ToolCall(
                    id="1", name="write_file", arguments={"path": "fizzbuzz.py", "content": "x"}
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    monkeypatch.setattr(base_agent_module, "call_with_tools", lambda *a, **k: responses.pop(0))
    _stub_turn_builders(monkeypatch)

    class _TesterLikeAgent(BaseAgent):
        agent_name = "tester"
        system_prompt = "test"
        tool_names = ["write_file"]

    agent = _TesterLikeAgent(config=_fake_config(tmp_path), tool_registry=reg)
    result = agent.run("write tests, but try to write to fizzbuzz.py instead")

    assert written["hit"] is False  # the real tool function never ran
    assert result.stopped_reason == "done"  # the agent loop itself didn't crash


def test_malformed_arguments_get_an_actionable_error_not_a_confusing_typeerror(
    monkeypatch, tmp_path
):
    """Seen live with gpt-oss-120b on OpenRouter: a tool call with broken
    JSON arguments used to surface as 'write_file() got an unexpected
    keyword argument _malformed_arguments' — technically informative, but
    indirect enough that the model gave up instead of retrying. This
    should now get an explicit instruction to retry with valid JSON, and
    the real tool function must never be called with the bogus kwarg."""
    called = {"hit": False}

    def fake_write_file(path, content):
        called["hit"] = True
        return "should never run"

    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="write_file",
            description="Writes a file.",
            input_schema={"type": "object", "properties": {}},
            fn=fake_write_file,
        )
    )

    responses = [
        LLMResponse(
            text=None,
            tool_calls=[
                ToolCall(
                    id="1", name="write_file", arguments={"_malformed_arguments": "{not valid"}
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    monkeypatch.setattr(base_agent_module, "call_with_tools", lambda *a, **k: responses.pop(0))
    _stub_turn_builders(monkeypatch)

    class _SomeAgent(BaseAgent):
        system_prompt = "test"
        tool_names = ["write_file"]

    agent = _SomeAgent(config=_fake_config(tmp_path), tool_registry=reg)
    result = agent.run("write a file")

    assert called["hit"] is False
    assert result.stopped_reason == "done"


def test_malformed_arguments_error_message_is_actionable(tmp_path):
    """Direct check on _execute_tool's output (not just that the loop
    survives) — the message must actually tell the model what to do."""
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="write_file",
            description="Writes a file.",
            input_schema={"type": "object", "properties": {}},
            fn=lambda **kwargs: "should never run",
        )
    )

    class _SomeAgent(BaseAgent):
        system_prompt = "test"
        tool_names = ["write_file"]

    agent = _SomeAgent(config=_fake_config(tmp_path), tool_registry=reg)
    agent.tracer = TraceLogger(task_id="t-malformed", config=agent.config)

    call = ToolCall(id="1", name="write_file", arguments={"_malformed_arguments": "{not valid"})
    result = agent._execute_tool(call, step=1)

    assert result.is_error is True
    assert "valid json" in result.output.lower()
    assert "retry" in result.output.lower()
