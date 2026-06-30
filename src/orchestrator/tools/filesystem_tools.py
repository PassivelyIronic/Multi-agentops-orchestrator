"""
Filesystem tools, sandboxed to a single root directory.

Every path is resolved and checked against the sandbox root before any read
or write happens. This is deliberately here in Phase 1, ahead of the formal
guardrails module, because these tools touch the real filesystem the moment
an agent runs — not a property we want to defer. Full guardrails
(configurable allow/deny lists, audit logging) land in Phase 2; this is the
non-negotiable minimum until then.

Uses get_active_config() (context.py), not get_config() directly, so an
isolated Config built for testing or eval purposes is actually respected
instead of silently falling back to the global env-based config.
"""

from __future__ import annotations

from pathlib import Path

from ..context import get_active_config
from .registry import tool

_MAX_READ_CHARS = 200_000  # guard against accidentally dumping a huge file into context


def _sandbox_root() -> Path:
    root = Path(get_active_config().sandbox_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_in_sandbox(relative_path: str) -> Path:
    root = _sandbox_root()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path escapes sandbox: {relative_path!r}")
    return candidate


@tool(
    name="read_file",
    description="Read a text file's contents, relative to the sandbox root.",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Relative file path"}},
        "required": ["path"],
    },
)
def read_file(path: str) -> str:
    target = _resolve_in_sandbox(path)
    if not target.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    data = target.read_text(encoding="utf-8", errors="replace")
    if len(data) > _MAX_READ_CHARS:
        return data[:_MAX_READ_CHARS] + "\n...[truncated]"
    return data


@tool(
    name="write_file",
    description=(
        "Write text content to a file, relative to the sandbox root. "
        "Creates parent directories as needed. Overwrites if the file exists."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative file path"},
            "content": {"type": "string", "description": "Text content to write"},
        },
        "required": ["path", "content"],
    },
)
def write_file(path: str, content: str) -> str:
    target = _resolve_in_sandbox(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {path}"


@tool(
    name="list_dir",
    description="List files and directories at a path relative to the sandbox root.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative directory path, '.' for the sandbox root",
            }
        },
        "required": ["path"],
    },
)
def list_dir(path: str) -> str:
    target = _resolve_in_sandbox(path)
    if not target.is_dir():
        raise NotADirectoryError(f"No such directory: {path}")
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return "\n".join(entries) if entries else "(empty)"
