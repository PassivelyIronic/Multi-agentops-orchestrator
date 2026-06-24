"""
Exec tool: run a shell command inside the sandbox with a hard timeout.

Intentionally narrow for Phase 1:
  - shell=False, args parsed with shlex.split — no shell injection via
    ';', '&&', backticks, pipes, etc.
  - cwd is always the sandbox directory.
  - a timeout always applies; subprocess.TimeoutExpired propagates up to
    the agent loop, which turns it into an error message back to the model
    instead of crashing the task.

A configurable command allow/deny list is a Phase 2 guardrails concern —
this just guarantees one run_command call can't hang forever or wander
outside the sandbox.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from ..config import get_config
from .registry import tool

_MAX_OUTPUT_CHARS = 20_000


@tool(
    name="run_command",
    description=(
        "Run a shell command inside the project sandbox and return its exit code, "
        "stdout, and stderr. Use this to run tests or inspect the project — not to "
        "install software or access the network."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command, e.g. 'pytest -v'"}
        },
        "required": ["command"],
    },
)
def run_command(command: str) -> str:
    cfg = get_config()
    sandbox = Path(cfg.sandbox_dir).resolve()
    sandbox.mkdir(parents=True, exist_ok=True)

    try:
        args = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"Could not parse command: {exc}") from exc
    if not args:
        raise ValueError("Empty command")

    result = subprocess.run(
        args,
        cwd=sandbox,
        capture_output=True,
        text=True,
        timeout=cfg.tool_exec_timeout_seconds,
        shell=False,
    )

    output = (
        f"exit_code={result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + "\n...[truncated]"
    return output
