"""
Guardrails: validates agent actions before execution.

This is a policy layer the agent loop consults before calling a tool —
separate from the tools' own hardcoded defenses (sandbox path resolution in
filesystem_tools.py, shell=False in exec_tools.py). Those stay fixed
properties of the tools. Guardrails are the *configurable* policy on top:
which commands are denied, how big a single write can be, which paths a
given agent role may write to — the kind of thing you'd want to tighten or
loosen per deployment without touching tool code.

A violation here means the tool function is never even called — the agent
loop gets an error ToolResult back immediately, and the block is recorded
in the trace exactly like a normal tool call would be (see tracing.py).

Phase 4 addition: role-scoped write restrictions. Tester and SWE share the
same tool *names* (read_file, write_file, run_command) — what makes them
different roles is what they're allowed to do with those tools, not the
tools themselves. Restricting Tester to test_*.py / *_test.py and PM to
*.md is enforced here, not just suggested in a system prompt — a model
ignoring its instructions still can't write to the wrong kind of file.

Known limitation: command matching is regex-over-the-raw-string, not an
AST-level shell parse. `\bgit push\b` blocks `git push origin main` (good)
but would also flag `echo "you should never git push --force"` (a harmless
string). That's a deliberate tradeoff for Phase 2 — a false positive that
blocks a benign echo is preferable to a false negative that lets a real
`git push --force` through. Tighter matching is a reasonable follow-up if
false positives turn out to matter in practice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import Config, get_config

DEFAULT_DENIED_COMMAND_PATTERNS: tuple[str, ...] = (
    r"\brm\b",
    r"\bdel\b",
    r"\brmdir\b",
    r"\bformat\b",
    r"\bshutdown\b",
    r"\bmkfs\b",
    r"\bdd\b",
    r"\bcurl\b",
    r"\bwget\b",
    r"\binvoke-webrequest\b",
    r"\bpip\s+install\b",
    r"\bconda\s+install\b",
    r"\bnpm\s+install\b",
    r"\bgit\s+push\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bsudo\b",
)

# Cloud instance-metadata endpoints — a health_check hitting one of these
# would leak credentials/role info from whatever machine the agent runs
# on. Not a full SSRF defense (no private-IP-range check, no redirect
# following check), but a cheap, high-value block against the single most
# common SSRF target.
DEFAULT_DENIED_HEALTH_CHECK_HOSTS: tuple[str, ...] = (
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.azure.com",
)


@dataclass
class GuardrailViolation:
    reason: str


def check(
    tool_name: str,
    arguments: dict[str, Any],
    config: Config | None = None,
    agent_name: str | None = None,
) -> GuardrailViolation | None:
    """Return a GuardrailViolation if this call should be blocked, else None."""
    cfg = config or get_config()

    if tool_name == "run_command":
        command = str(arguments.get("command", ""))
        for pattern in cfg.denied_command_patterns or DEFAULT_DENIED_COMMAND_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return GuardrailViolation(
                    f"Command blocked by guardrail policy (matched {pattern!r}): {command!r}"
                )

    if tool_name == "write_file":
        content = str(arguments.get("content", ""))
        size = len(content.encode("utf-8"))
        if size > cfg.max_write_bytes:
            return GuardrailViolation(
                f"Write blocked: {size} bytes exceeds MAX_WRITE_BYTES={cfg.max_write_bytes}"
            )

        path = str(arguments.get("path", ""))
        if agent_name == "tester" and not _looks_like_test_file(path):
            return GuardrailViolation(
                f"Tester agent may only write test files (test_*.py or *_test.py), got: {path!r}"
            )
        if agent_name == "pm" and not path.lower().endswith(".md"):
            return GuardrailViolation(
                f"PM agent may only write markdown files (.md), got: {path!r}"
            )

    if tool_name == "health_check":
        url = str(arguments.get("url", ""))
        for host in DEFAULT_DENIED_HEALTH_CHECK_HOSTS:
            if host in url:
                return GuardrailViolation(
                    f"health_check blocked: target host is a known SSRF-sensitive endpoint ({host})"
                )

    return None


def _looks_like_test_file(path: str) -> bool:
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return (name.startswith("test_") and name.endswith(".py")) or name.endswith("_test.py")
