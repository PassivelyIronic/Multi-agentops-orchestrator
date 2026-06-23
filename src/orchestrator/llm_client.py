"""
Provider-agnostic LLM client with tool-use support.

This is the only module that imports the Gemini or Anthropic SDKs directly.
Every agent talks to `call_with_tools()` and gets back the same response
shape regardless of which provider is configured — that's the whole reason
this file exists, and it's what lets development happen for free on a
Gemini key while Anthropic gets used selectively (e.g. evaluation runs).

Tool schemas passed in here use Anthropic's shape — {"name", "description",
"input_schema"} — since it's the simplest superset. When targeting Gemini,
`_call_gemini` translates it into a FunctionDeclaration internally, so
agent code never has to special-case the provider.

Note: both Gemini's `google-genai` and Anthropic's SDK move fast. If a call
here breaks, check the current docs before assuming the logic is wrong —
this was last verified against google-genai's documented FunctionDeclaration
/ GenerateContentConfig pattern and Anthropic's messages.create tool_use
pattern in June 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Config, get_config


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized response shape — identical regardless of provider."""

    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None


def call_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    system: str | None = None,
    config: Config | None = None,
) -> LLMResponse:
    """Send one chat turn to whichever provider is configured."""
    cfg = config or get_config()

    if cfg.llm_provider == "gemini":
        return _call_gemini(messages, tools, system, cfg)
    if cfg.llm_provider == "anthropic":
        return _call_anthropic(messages, tools, system, cfg)
    raise ValueError(f"Unknown LLM_PROVIDER: {cfg.llm_provider!r}")


def _call_anthropic(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    system: str | None,
    cfg: Config,
) -> LLMResponse:
    import anthropic

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    response = client.messages.create(
        model=cfg.anthropic_model,
        max_tokens=4096,
        system=system or "",
        messages=messages,
        tools=tools,
    )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

    return LLMResponse(
        text="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        raw=response,
    )


def _call_gemini(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    system: str | None,
    cfg: Config,
) -> LLMResponse:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=cfg.gemini_api_key)

    declarations = [
        types.FunctionDeclaration(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters_json_schema=tool.get("input_schema", {"type": "object", "properties": {}}),
        )
        for tool in tools
    ]
    gemini_tools = [types.Tool(function_declarations=declarations)] if declarations else None

    # NOTE: this assumes simple {"role", "content": str} turns. Feeding a tool
    # *result* back in needs a richer Part (function_response) — that lands in
    # Phase 1 once the agent loop actually executes tools and replies with output.
    contents = [
        types.Content(
            role=_to_gemini_role(m["role"]), parts=[types.Part.from_text(text=m["content"])]
        )
        for m in messages
        if isinstance(m["content"], str)
    ]

    response = client.models.generate_content(
        model=cfg.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            tools=gemini_tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

    tool_calls = [
        ToolCall(id=f"gemini_{i}", name=fc.name, arguments=dict(fc.args))
        for i, fc in enumerate(response.function_calls or [])
    ]

    usage = response.usage_metadata
    return LLMResponse(
        text=response.text if not tool_calls else None,
        tool_calls=tool_calls,
        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        raw=response,
    )


def _to_gemini_role(role: str) -> str:
    """Gemini uses 'model' where Anthropic/OpenAI use 'assistant'."""
    return "model" if role == "assistant" else "user"
