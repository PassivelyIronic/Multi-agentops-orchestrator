"""
Tool registry: maps tool name -> (schema, callable).

Agents declare which tool *names* they're allowed to use; the registry is
the single source of truth both for the JSON schemas sent to the LLM and
for the actual Python functions that run when the model calls them.

`ToolRegistry` is a plain class rather than a bag of module functions so
tests can build an isolated registry instead of mutating the shared one.
The `@tool` decorator below registers into a single shared `registry`
instance, which is what concrete agents use in normal operation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def schemas_for(self, names: list[str]) -> list[dict[str, Any]]:
        """Anthropic-shaped schemas — llm_client translates these for Gemini."""
        missing = [n for n in names if n not in self._tools]
        if missing:
            raise KeyError(f"Unknown tool name(s): {missing}")
        return [
            {
                "name": self._tools[n].name,
                "description": self._tools[n].description,
                "input_schema": self._tools[n].input_schema,
            }
            for n in names
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Run a tool by name and return its output as a string.

        Raises on unknown tool name or on whatever the tool function itself
        raises (e.g. FileNotFoundError, subprocess.TimeoutExpired). Callers
        — the agent loop — are responsible for catching exceptions here and
        turning them into an error ToolResult instead of letting them crash
        the task; the registry's job is just to run the function honestly.
        """
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].fn(**arguments)


# Shared instance that concrete agents' tool modules register into at
# import time via the @tool decorator below.
registry = ToolRegistry()


def tool(name: str, description: str, input_schema: dict[str, Any]):
    """Decorator: registers a function as a tool under `name` in the shared registry."""

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        registry.register(
            ToolSpec(name=name, description=description, input_schema=input_schema, fn=fn)
        )
        return fn

    return decorator
