"""
Unit tests for context.py — the contextvar-based config threading that
fixes a real bug: tool functions used to call get_config() directly,
silently ignoring whatever Config the calling agent actually had.
"""

from __future__ import annotations

from orchestrator.config import Config
from orchestrator.context import get_active_config, use_config


def _fake_config(**overrides) -> Config:
    defaults = dict(llm_provider="gemini", gemini_api_key="fake", anthropic_api_key="fake")
    defaults.update(overrides)
    return Config(**defaults)


def test_falls_back_to_global_config_when_nothing_set(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("SANDBOX_DIR", "/some/global/path")

    assert get_active_config().sandbox_dir == "/some/global/path"


def test_use_config_makes_that_config_active_inside_the_block():
    cfg = _fake_config(sandbox_dir="/isolated/path")

    with use_config(cfg):
        assert get_active_config() is cfg


def test_use_config_reverts_after_the_block(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    cfg = _fake_config(sandbox_dir="/isolated/path")

    with use_config(cfg):
        pass

    assert get_active_config() is not cfg


def test_use_config_supports_nesting():
    outer = _fake_config(sandbox_dir="/outer")
    inner = _fake_config(sandbox_dir="/inner")

    with use_config(outer):
        assert get_active_config() is outer
        with use_config(inner):
            assert get_active_config() is inner
        assert get_active_config() is outer  # restored, not lost


def test_use_config_resets_even_if_the_block_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    cfg = _fake_config(sandbox_dir="/isolated/path")

    try:
        with use_config(cfg):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert get_active_config() is not cfg
