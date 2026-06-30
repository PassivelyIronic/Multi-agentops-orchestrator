"""
LLM-as-judge scoring for golden-dataset tasks whose success criteria are
qualitative ("is this backlog well-prioritized?") rather than mechanically
checkable (file exists, command exits 0).

Reuses call_with_tools with an empty tool list — the same pattern
orchestrator.py's decomposition call uses — so this judge call gets
retries/backoff for free instead of needing its own LLM-calling path.
"""

from __future__ import annotations

import json

from .config import Config, get_config
from .llm_client import call_with_tools

JUDGE_SYSTEM_PROMPT = """\
You are an evaluation judge. You are given a rubric describing what a \
correct result should contain, and the actual content to judge against it.

Respond with ONLY a JSON object, no prose, no markdown fences:
{"passed": true or false, "reasoning": "<one or two sentences>"}
"""


def judge(rubric: str, context: str, config: Config | None = None) -> tuple[bool, str]:
    """Returns (passed, reasoning). Never raises — a judge call that fails
    to parse or errors out counts as a failed verification (fail-closed),
    with the reasoning explaining why, rather than crashing the eval run."""
    cfg = config or get_config()
    prompt = f"Rubric: {rubric}\n\nContent to judge:\n{context}"
    try:
        response = call_with_tools(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            system=JUDGE_SYSTEM_PROMPT,
            config=cfg,
        )
    except Exception as exc:
        return False, f"Judge call failed: {exc}"
    return _parse_verdict(response.text)


def _parse_verdict(raw_text: str | None) -> tuple[bool, str]:
    if not raw_text:
        return False, "Judge returned no response."

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return False, f"Judge response was not valid JSON: {raw_text[:200]!r}"

    if not isinstance(data, dict) or "passed" not in data:
        return False, f"Judge response missing 'passed' field: {raw_text[:200]!r}"

    return bool(data["passed"]), str(data.get("reasoning", ""))
