"""
Guardrails: validates agent actions before execution.

This is a policy layer the agent loop consults before calling a tool —
separate from the tools' own hardcoded defenses (sandbox path resolution in
filesystem_tools.py, shell=False in exec_tools.py). Those stay fixed
properties of the tools. Guardrails are the *configurable* policy on top:
which commands are denied, how big a single write can be — the kind of
thing you'd want to tighten or loosen per deployment without touching tool
code.

A violation here means the tool function is never even called — the agent
loop gets an error ToolResult back immediately, and the block is recorded
in the trace exactly like a normal tool call would be (see tracing.py).

Known limitation: command matching is regex-over-the-raw-string, not an
AST-level shell parse. `\bgit push\b` blocks `git push origin main` (good)
but would also flag `echo "you should never git push --force"` (a harmless
string). That's a deliberate tradeoff for Phase 2 — a false positive that
blocks a benign echo is preferable to a false negative that lets a real
`git push --force` through. Tighter matching is a reasonable Phase 2.5
follow-up if false positives turn out to matter in practice.
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


@dataclass
class GuardrailViolation:
    reason: str


def check(
    tool_name: str, arguments: dict[str, Any], config: Config | None = None
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

    return None
