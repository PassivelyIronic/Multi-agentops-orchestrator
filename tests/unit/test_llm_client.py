"""
Unit tests for the provider-routing logic in llm_client.py.

These do NOT call real APIs — they patch the provider-specific functions and
check that call_with_tools() routes to the right one. Real API smoke tests
belong in tests/integration, once Phase 1 lands actual tool execution.
"""

import pytest

from orchestrator import llm_client
from orchestrator.config import Config


def _fake_config(provider: str) -> Config:
    return Config(
        llm_provider=provider,
        gemini_api_key="fake",
        anthropic_api_key="fake",
        gemini_model="gemini-2.5-flash",
        anthropic_model="claude-sonnet-4-6",
        max_steps_per_agent=15,
        db_path=":memory:",
    )


def test_routes_to_gemini(monkeypatch):
    called = {}
    monkeypatch.setattr(llm_client, "_call_gemini", lambda *a, **k: called.setdefault("hit", True))

    llm_client.call_with_tools([], [], config=_fake_config("gemini"))

    assert called.get("hit") is True


def test_routes_to_anthropic(monkeypatch):
    called = {}
    monkeypatch.setattr(
        llm_client, "_call_anthropic", lambda *a, **k: called.setdefault("hit", True)
    )

    llm_client.call_with_tools([], [], config=_fake_config("anthropic"))

    assert called.get("hit") is True


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        llm_client.call_with_tools([], [], config=_fake_config("made-up"))
