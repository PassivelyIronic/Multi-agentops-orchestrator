"""
Web search for the PM agent's research step.

Uses DuckDuckGo's keyless Instant Answer API — no API key or paid
subscription needed, which matters for a portfolio project with no search
API budget. Trade-off: it returns an abstract/summary, not a ranked list of
results like a real search engine — good enough for "what's the current
state of X" background research before breaking a requirement down, not a
substitute for a full SERP. Swapping in a real search API (Google Custom
Search, Bing, Tavily, etc.) would be a natural upgrade if this turns out to
be too shallow in practice.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from .registry import tool

_TIMEOUT_SECONDS = 10
_MAX_RELATED_TOPICS = 5


@tool(
    name="web_search",
    description=(
        "Search the web for background on a topic. Returns a brief summary and a "
        "few related points, not a full ranked list of results."
    ),
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    },
)
def web_search(query: str) -> str:
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
    )
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return f"Search failed: {exc}"

    abstract = data.get("AbstractText") or ""
    related = [
        topic.get("Text", "")
        for topic in data.get("RelatedTopics", [])
        if isinstance(topic, dict) and topic.get("Text")
    ][:_MAX_RELATED_TOPICS]

    if not abstract and not related:
        return (
            f"No summary available for {query!r}. Try a more specific or differently-phrased query."
        )

    parts = []
    if abstract:
        parts.append(abstract)
    if related:
        parts.append("Related: " + "; ".join(related))
    return "\n".join(parts)
