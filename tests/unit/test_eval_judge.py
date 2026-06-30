"""
Unit tests for eval_judge.judge() — call_with_tools is mocked, so these
never make a real API call. Mirrors the defensive-parsing style already
used for orchestrator._parse_subtasks (markdown-fence stripping, fallback
on malformed JSON), since this is the same kind of "ask the model for
structured output, degrade gracefully if it doesn't comply" problem.
"""

from __future__ import annotations

import json

from orchestrator import eval_judge
from orchestrator.config import Config
from orchestrator.llm_client import LLMResponse


def _fake_config(**overrides) -> Config:
    defaults = dict(llm_provider="gemini", gemini_api_key="fake", anthropic_api_key="fake")
    defaults.update(overrides)
    return Config(**defaults)


def test_judge_parses_pass_verdict(monkeypatch):
    monkeypatch.setattr(
        eval_judge,
        "call_with_tools",
        lambda *a, **k: LLMResponse(text=json.dumps({"passed": True, "reasoning": "looks good"})),
    )

    passed, reasoning = eval_judge.judge("some rubric", "some content", _fake_config())

    assert passed is True
    assert reasoning == "looks good"


def test_judge_parses_fail_verdict(monkeypatch):
    monkeypatch.setattr(
        eval_judge,
        "call_with_tools",
        lambda *a, **k: LLMResponse(
            text=json.dumps({"passed": False, "reasoning": "missing items"})
        ),
    )

    passed, reasoning = eval_judge.judge("some rubric", "some content", _fake_config())

    assert passed is False
    assert reasoning == "missing items"


def test_judge_strips_markdown_fences():
    raw = "```json\n" + json.dumps({"passed": True, "reasoning": "ok"}) + "\n```"
    assert eval_judge._parse_verdict(raw) == (True, "ok")


def test_judge_fails_closed_on_malformed_json():
    passed, reasoning = eval_judge._parse_verdict("not json at all")
    assert passed is False
    assert "not valid json" in reasoning.lower()


def test_judge_fails_closed_on_empty_response():
    passed, reasoning = eval_judge._parse_verdict(None)
    assert passed is False


def test_judge_fails_closed_when_passed_field_missing():
    passed, reasoning = eval_judge._parse_verdict(json.dumps({"reasoning": "no passed field"}))
    assert passed is False


def test_judge_fails_closed_when_llm_call_raises(monkeypatch):
    monkeypatch.setattr(
        eval_judge, "call_with_tools", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("429"))
    )

    passed, reasoning = eval_judge.judge("rubric", "content", _fake_config())

    assert passed is False
    assert "429" in reasoning
