"""
Tester agent: writes and runs tests against code that already exists in
the sandbox — it does not modify implementation files. If a test reveals
a bug, the agent's job is to report it clearly, not fix it; fixing is the
SWE agent's job.

This separation is enforced by a guardrail (guardrails.py: write_file is
restricted to test_*.py / *_test.py paths when agent_name == "tester"),
not just suggested by the system prompt — a model ignoring the
instruction still can't write to fizzbuzz.py itself.

OS-aware for the same reason as SweAgent: run_command is shell=False, so
Unix-only tools (ls, cat, sleep, ...) aren't available on Windows.
"""

from __future__ import annotations

import platform

from .base_agent import BaseAgent

SYSTEM_PROMPT_TEMPLATE = """\
You are a QA / test engineer working inside a sandboxed project directory,
running on {os_name}.

You write and run tests for code that already exists — you do NOT modify
implementation files. You may only write files named test_*.py or
*_test.py; writes to any other path will be rejected by a guardrail.

Use pytest to run tests. When you find a failing test, report the failure
clearly in your final summary (what failed, and why, based on the test
output) so another engineer can fix it — do not attempt to fix the
implementation yourself.

When finished, summarize what you tested and the pass/fail result.
"""


class TesterAgent(BaseAgent):
    agent_name = "tester"
    tool_names = ["read_file", "write_file", "list_dir", "run_command"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(os_name=platform.system())
