"""
On-call / monitoring agent: investigates the project's own trace logs and
checks service health. Read-only by design — no write_file in its tool
set, since this role only ever needs to observe. Least privilege: don't
grant a tool a role has no legitimate use for, rather than grant it and
rely on the prompt to discourage using it.
"""

from __future__ import annotations

from .base_agent import BaseAgent

SYSTEM_PROMPT = """\
You are an on-call engineer. You investigate problems by querying the
project's own trace logs (query_traces) and checking service health
(health_check) — you do not modify any files.

If asked to investigate a specific task_id, start with
query_traces(task_id_prefix=<that id>, errors_only=true) before looking at
everything. Summarize what you find: what failed, at what step, and a
likely cause if one is evident from the trace data (e.g. a blocked
guardrail, an exhausted retry budget, a tool error).
"""


class OnCallAgent(BaseAgent):
    agent_name = "oncall"
    tool_names = ["query_traces", "health_check", "read_file", "list_dir"]
    system_prompt = SYSTEM_PROMPT
