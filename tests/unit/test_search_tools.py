"""
Unit tests for search_tools.py — urllib.request.urlopen is mocked so these
never make a real network call to DuckDuckGo.
"""

from __future__ import annotations

import json
from io import BytesIO

from orchestrator.tools import search_tools
from orchestrator.tools.search_tools import web_search


def _fake_response(payload: dict):
    class _FakeResponse(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return _FakeResponse(json.dumps(payload).encode("utf-8"))


def test_web_search_returns_abstract(monkeypatch):
    monkeypatch.setattr(
        search_tools.urllib.request,
        "urlopen",
        lambda *a, **k: _fake_response({"AbstractText": "A summary.", "RelatedTopics": []}),
    )

    result = web_search("some topic")

    assert "A summary." in result


def test_web_search_includes_related_topics(monkeypatch):
    monkeypatch.setattr(
        search_tools.urllib.request,
        "urlopen",
        lambda *a, **k: _fake_response(
            {
                "AbstractText": "",
                "RelatedTopics": [{"Text": "Related point one"}, {"Text": "Related point two"}],
            }
        ),
    )

    result = web_search("some topic")

    assert "Related point one" in result
    assert "Related point two" in result


def test_web_search_caps_related_topics_at_five(monkeypatch):
    topics = [{"Text": f"point {i}"} for i in range(10)]
    monkeypatch.setattr(
        search_tools.urllib.request,
        "urlopen",
        lambda *a, **k: _fake_response({"AbstractText": "", "RelatedTopics": topics}),
    )

    result = web_search("some topic")

    assert result.count("point") == 5


def test_web_search_reports_no_summary_when_empty(monkeypatch):
    monkeypatch.setattr(
        search_tools.urllib.request,
        "urlopen",
        lambda *a, **k: _fake_response({"AbstractText": "", "RelatedTopics": []}),
    )

    result = web_search("an obscure query")

    assert "no summary" in result.lower()


def test_web_search_handles_network_failure_gracefully(monkeypatch):
    def _raise(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(search_tools.urllib.request, "urlopen", _raise)

    result = web_search("anything")

    assert "search failed" in result.lower()
