"""
Unit tests for the retry/backoff helper in llm_client.py — the piece that
makes call_with_tools resilient to rate limits (HTTP 429) and transient
server errors (5xx) from either provider, without retrying forever on
errors that won't resolve themselves (e.g. a plain bad-request 4xx).
"""

from __future__ import annotations

import pytest

from orchestrator.llm_client import _should_retry, _with_retries


class _FakeStatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_should_retry_on_429_and_5xx():
    assert _should_retry(_FakeStatusError(429)) is True
    assert _should_retry(_FakeStatusError(503)) is True


def test_should_not_retry_on_other_4xx():
    assert _should_retry(_FakeStatusError(400)) is False
    assert _should_retry(_FakeStatusError(404)) is False


def test_with_retries_succeeds_after_transient_failures():
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise _FakeStatusError(429)
        return "ok"

    result = _with_retries(flaky, max_retries=5, base_delay=0.0)

    assert result == "ok"
    assert calls["count"] == 3


def test_with_retries_gives_up_after_max_retries():
    def always_fails():
        raise _FakeStatusError(429)

    with pytest.raises(_FakeStatusError):
        _with_retries(always_fails, max_retries=2, base_delay=0.0)


def test_with_retries_does_not_retry_non_retryable_errors():
    calls = {"count": 0}

    def fails_hard():
        calls["count"] += 1
        raise ValueError("not a rate limit, just a bug")

    with pytest.raises(ValueError):
        _with_retries(fails_hard, max_retries=5, base_delay=0.0)

    assert calls["count"] == 1  # gave up immediately, no retries spent
