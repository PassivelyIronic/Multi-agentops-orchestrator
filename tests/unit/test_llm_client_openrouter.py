"""
Deeper tests for the OpenRouter integration: response parsing (including
malformed tool-call JSON from a weaker free model) and the turn-builder
functions across providers. openai.OpenAI is faked rather than hitting the
network, but the real parsing logic inside _call_openrouter still runs
against the scripted response — this is what actually exercises the
json.loads / tool_calls handling, not just routing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from orchestrator.config import Config
from orchestrator.llm_client import (
    ToolResult,
    _call_openrouter,
    to_assistant_turn,
    to_tool_result_turn,
)


def _fake_config(**overrides) -> Config:
    defaults = dict(
        llm_provider="openrouter",
        gemini_api_key="fake",
        anthropic_api_key="fake",
        openrouter_api_key="fake",
    )
    defaults.update(overrides)
    return Config(**defaults)


def _fake_openai_response(content=None, tool_calls=None, prompt_tokens=10, completion_tokens=5):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _patch_openai_client(monkeypatch, response):
    """Fakes openai.OpenAI(...) so _call_openrouter's real parsing logic
    runs against a scripted response instead of a real HTTP call."""

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **k: response))

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)


# --- _call_openrouter parsing --------------------------------------------


def test_call_openrouter_parses_text_response(monkeypatch):
    response = _fake_openai_response(content="hello there")
    _patch_openai_client(monkeypatch, response)

    result = _call_openrouter([], [], system=None, cfg=_fake_config())

    assert result.text == "hello there"
    assert result.tool_calls == []
    assert result.input_tokens == 10
    assert result.output_tokens == 5


def test_call_openrouter_parses_valid_tool_call(monkeypatch):
    tool_call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="echo", arguments=json.dumps({"text": "hi"}))
    )
    response = _fake_openai_response(content=None, tool_calls=[tool_call])
    _patch_openai_client(monkeypatch, response)

    result = _call_openrouter([], [], system=None, cfg=_fake_config())

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "echo"
    assert result.tool_calls[0].arguments == {"text": "hi"}
    assert result.text is None  # text is suppressed whenever there are tool calls


def test_call_openrouter_handles_malformed_tool_call_json(monkeypatch):
    """A weaker free model returns broken JSON in tool call arguments — this
    must not raise; it degrades to a marked ToolCall that
    base_agent._execute_tool's existing error isolation turns into a normal
    error result instead of crashing the task."""
    tool_call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="echo", arguments="{not valid json")
    )
    response = _fake_openai_response(content=None, tool_calls=[tool_call])
    _patch_openai_client(monkeypatch, response)

    result = _call_openrouter([], [], system=None, cfg=_fake_config())

    assert len(result.tool_calls) == 1
    assert "_malformed_arguments" in result.tool_calls[0].arguments


def test_call_openrouter_handles_empty_arguments_string(monkeypatch):
    tool_call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="list_dir", arguments="")
    )
    response = _fake_openai_response(content=None, tool_calls=[tool_call])
    _patch_openai_client(monkeypatch, response)

    result = _call_openrouter([], [], system=None, cfg=_fake_config())

    assert result.tool_calls[0].arguments == {}


# --- turn builders ---------------------------------------------------------


def test_to_assistant_turn_anthropic_is_single_element_list():
    response = SimpleNamespace(raw=SimpleNamespace(content=["block"]))
    result = to_assistant_turn(response, _fake_config(llm_provider="anthropic"))
    assert result == [{"role": "assistant", "content": ["block"]}]


def test_to_assistant_turn_openrouter_includes_tool_calls():
    tool_call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="echo", arguments='{"text": "hi"}')
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    response = SimpleNamespace(raw=SimpleNamespace(choices=[SimpleNamespace(message=message)]))

    result = to_assistant_turn(response, _fake_config())

    assert result == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "hi"}'},
                }
            ],
        }
    ]


def test_to_tool_result_turn_anthropic_bundles_into_one_turn():
    results = [
        ToolResult(tool_call_id="1", name="echo", output="hi", is_error=False),
        ToolResult(tool_call_id="2", name="boom", output="err", is_error=True),
    ]
    result = to_tool_result_turn(results, _fake_config(llm_provider="anthropic"))

    assert len(result) == 1  # one combined turn for both results
    assert result[0]["role"] == "user"
    assert len(result[0]["content"]) == 2


def test_to_tool_result_turn_openrouter_one_message_per_result():
    results = [
        ToolResult(tool_call_id="1", name="echo", output="hi", is_error=False),
        ToolResult(tool_call_id="2", name="boom", output="err", is_error=True),
    ]
    result = to_tool_result_turn(results, _fake_config())

    assert result == [
        {"role": "tool", "tool_call_id": "1", "content": "hi"},
        {"role": "tool", "tool_call_id": "2", "content": "err"},
    ]
