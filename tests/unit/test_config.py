"""Unit tests for config.py — no network calls, no real API keys needed."""

import pytest

from orchestrator.config import get_config


def test_defaults_to_gemini_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    cfg = get_config()

    assert cfg.llm_provider == "gemini"
    assert cfg.max_steps_per_agent == 15


def test_raises_when_provider_key_missing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        get_config()


def test_raises_on_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "made-up-provider")

    with pytest.raises(ValueError):
        get_config()
