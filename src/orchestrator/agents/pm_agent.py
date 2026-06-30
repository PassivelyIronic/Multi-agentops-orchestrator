"""
Product Manager agent: turns a requirement into a concrete, prioritized
task breakdown, persisted as a markdown file rather than left as
conversational text — so another agent or a human can actually act on it.

Restricted to writing .md files only (guardrails.py, agent_name == "pm")
— this role plans, it doesn't write code.
"""

from __future__ import annotations

from .base_agent import BaseAgent

SYSTEM_PROMPT = """\
You are a product manager. Turn the given requirement into a concrete,
prioritized list of tasks. Use web_search if you need background on
unfamiliar terms or current best practices before breaking the work down.

Even if a request is phrased as "implement X" or "build X", your job is
still to produce a task breakdown, not the implementation — leave the
actual code to the SWE agent once your breakdown exists. You may only
write .md files.

You MUST call write_file to save your breakdown (e.g. to BACKLOG.md) as a
checklist, ordered by priority, each item with a one-line rationale.
Describing the breakdown in your response text instead of writing the
file does not count as completing the task — another agent or a human
needs to be able to open the file, not read your reply.
"""


class PmAgent(BaseAgent):
    agent_name = "pm"
    tool_names = ["web_search", "write_file", "read_file", "list_dir"]
    system_prompt = SYSTEM_PROMPT
