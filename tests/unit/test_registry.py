"""
Unit tests for ToolRegistry — using a fresh, isolated instance rather than
the shared `registry` singleton, so these tests don't depend on (or
interfere with) whatever tools other modules have registered.
"""

from __future__ import annotations

import pytest

from orchestrator.tools.registry import ToolRegistry, ToolSpec


def _make_spec(name: str = "echo") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Echoes its input back.",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        fn=lambda text: f"echo: {text}",
    )


def test_register_and_execute():
    reg = ToolRegistry()
    reg.register(_make_spec())

    assert reg.execute("echo", {"text": "hi"}) == "echo: hi"


def test_schemas_for_returns_anthropic_shape():
    reg = ToolRegistry()
    reg.register(_make_spec())

    schemas = reg.schemas_for(["echo"])

    assert schemas == [
        {
            "name": "echo",
            "description": "Echoes its input back.",
            "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
        }
    ]


def test_duplicate_registration_raises():
    reg = ToolRegistry()
    reg.register(_make_spec())

    with pytest.raises(ValueError):
        reg.register(_make_spec())


def test_unknown_tool_execute_raises():
    reg = ToolRegistry()

    with pytest.raises(KeyError):
        reg.execute("does_not_exist", {})


def test_unknown_tool_in_schemas_for_raises():
    reg = ToolRegistry()
    reg.register(_make_spec())

    with pytest.raises(KeyError):
        reg.schemas_for(["echo", "does_not_exist"])
