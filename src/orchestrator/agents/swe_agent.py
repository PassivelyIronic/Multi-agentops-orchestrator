"""
Software Engineer agent: reads/writes files and runs commands inside the
sandbox to accomplish a coding task.

The system prompt is built per-instance, not as a fixed class-level string,
so it can tell the model which OS it's actually running on. run_command is
shell=False, so it never goes through cmd.exe or bash — whether a given
command works depends entirely on whether that program exists as a real
executable on the host OS. Most Unix coreutils (ls, cat, rm, grep, sleep)
aren't present on a bare Windows install, so the agent is steered toward
the sandboxed filesystem tools for basic file operations and toward
run_command mainly for things like running tests or python scripts —
which is also the more sandboxed choice, since read_file/write_file/
list_dir validate every path against the sandbox root explicitly, while a
shelled-out `ls` only relies on cwd confinement.

Tools intentionally exclude git for now — committing is exactly the kind of
action that should go through the guardrails module (Phase 2) before an
agent gets to do it autonomously. git_tools.py stays a stub until then.
"""

from __future__ import annotations

import platform

from .base_agent import BaseAgent

SYSTEM_PROMPT_TEMPLATE = """\
You are a software engineer working inside a sandboxed project directory,
running on {os_name}.

You have tools to read files, write files, list directories, and run shell
commands. Prefer read_file / write_file / list_dir over run_command for
basic file operations — they're sandboxed and portable across operating
systems. Reserve run_command for things like running tests (e.g. `pytest`)
or python scripts.

run_command does not go through a shell, so Unix-only tools (ls, cat, rm,
grep, sleep) are not available on Windows — use python or the filesystem
tools instead when running on Windows.

When the task is complete, respond with a short summary of what you changed
and stop calling tools.
"""


class SweAgent(BaseAgent):
    tool_names = ["read_file", "write_file", "list_dir", "run_command"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(os_name=platform.system())
