"""
Provider-agnostic LLM client with tool-use support.

This is the only module that imports the Gemini or Anthropic SDKs directly.
Every agent talks to `call_with_tools()` and gets back the same response
shape regardless of which provider is configured. Two more functions —
`to_assistant_turn` and `to_tool_result_turn` — build the next conversation
turn after a tool executes, again hiding the provider-specific shape from
the agent loop.

Resilience: call_with_tools wraps the actual provider call with a retry +
exponential backoff loop. This covers transient failures — HTTP 429 (rate
limit) and 5xx (server error) — which both Gemini and Anthropic return under
the same status-code conventions despite different SDK exception classes.
It deliberately does NOT retry forever: a 429 caused by "daily quota
exhausted" won't resolve itself by waiting a few seconds, so after
max_retries we give up and surface a clear error instead of hanging.

Note: both `google-genai` and the Anthropic SDK move fast. If a call here
breaks, check current docs before assuming the logic is wrong — this was
last verified against google-genai's documented FunctionDeclaration /
GenerateContentConfig / function_response pattern and Anthropic's
messages.create tool_use pattern in June 2026.
"""

from __future__ import annotations

import random
import time
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
class ToolResult:
    """The outcome of actually running a ToolCall, ready to send back."""

    tool_call_id: str
    name: str
    output: str
    is_error: bool = False


@dataclass
class LLMResponse:
    """Normalized response shape — identical regardless of provider."""

    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def call_with_tools(
    messages: list[Any],
    tools: list[dict[str, Any]],
    system: str | None = None,
    config: Config | None = None,
) -> LLMResponse:
    """
    Send one chat turn to whichever provider is configured, retrying on
    transient failures.

    `messages` items are either generic dicts {"role": "user"/"assistant",
    "content": str} for plain text turns, or the provider-native objects
    returned by `to_assistant_turn` / `to_tool_result_turn` for turns that
    carry tool calls/results. Both `_call_anthropic` and `_call_gemini`
    accept this mix.

    `tools` uses Anthropic's schema shape — {"name", "description",
    "input_schema"} — since it's the simplest superset; `_call_gemini`
    translates it into a FunctionDeclaration internally.
    """
    cfg = config or get_config()

    if cfg.llm_provider == "gemini":
        return _with_retries(
            lambda: _call_gemini(messages, tools, system, cfg),
            max_retries=cfg.max_retries,
            base_delay=cfg.retry_base_delay_seconds,
        )
    if cfg.llm_provider == "anthropic":
        return _with_retries(
            lambda: _call_anthropic(messages, tools, system, cfg),
            max_retries=cfg.max_retries,
            base_delay=cfg.retry_base_delay_seconds,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {cfg.llm_provider!r}")


def to_assistant_turn(response: LLMResponse, config: Config) -> Any:
    """
    Build the assistant turn to append to history after a response, in the
    shape the configured provider expects to see its own prior turn.
    """
    if config.llm_provider == "anthropic":
        return {"role": "assistant", "content": response.raw.content}
    if config.llm_provider == "gemini":
        return response.raw.candidates[0].content
    raise ValueError(f"Unknown LLM_PROVIDER: {config.llm_provider!r}")


def to_tool_result_turn(results: list[ToolResult], config: Config) -> Any:
    """Build the next turn carrying tool execution results back to the model."""
    if config.llm_provider == "anthropic":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": r.tool_call_id,
                    "content": r.output,
                    **({"is_error": True} if r.is_error else {}),
                }
                for r in results
            ],
        }
    if config.llm_provider == "gemini":
        from google.genai import types

        parts = [
            types.Part.from_function_response(
                name=r.name,
                response={"error": r.output} if r.is_error else {"result": r.output},
            )
            for r in results
        ]
        return types.Content(role="tool", parts=parts)
    raise ValueError(f"Unknown LLM_PROVIDER: {config.llm_provider!r}")


# --------------------------------------------------------------------------
# Retry / backoff
# --------------------------------------------------------------------------


def _with_retries(fn, max_retries: int, base_delay: float):
    """
    Call fn(), retrying with exponential backoff + jitter on retryable
    errors. Re-raises immediately on non-retryable errors, and re-raises
    the last error once max_retries is exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries or not _should_retry(exc):
                raise
            last_exc = exc
            delay = base_delay * (2**attempt) + random.uniform(0, base_delay)
            time.sleep(delay)
    raise last_exc  # pragma: no cover - unreachable, loop always returns or raises


def _should_retry(exc: Exception) -> bool:
    """
    Decide whether an exception from either provider's SDK is worth
    retrying. Both SDKs expose an HTTP-style status code on rate-limit /
    server errors under slightly different attribute names, so we check the
    common ones by duck typing instead of importing both SDKs' exception
    hierarchies here.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    if isinstance(status, int):
        return status in (429, 500, 502, 503, 504)

    # No status code available — fall back to a name-based guess for
    # connection/timeout-shaped errors, which are also worth one retry.
    name = exc.__class__.__name__.lower()
    return "timeout" in name or "connection" in name


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


def _call_anthropic(
    messages: list[Any],
    tools: list[dict[str, Any]],
    system: str | None,
    cfg: Config,
) -> LLMResponse:
    import anthropic

    client = anthropic.Anthropic(
        api_key=cfg.anthropic_api_key, timeout=cfg.llm_request_timeout_seconds
    )

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


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------


def _call_gemini(
    messages: list[Any],
    tools: list[dict[str, Any]],
    system: str | None,
    cfg: Config,
) -> LLMResponse:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=cfg.gemini_api_key)
    # NOTE: no explicit per-call timeout here. google-genai's HttpOptions
    # exposes a timeout setting, but its exact units/behavior weren't
    # confirmed against current docs at the time of writing — rather than
    # guess, we rely on the task-level wall-clock timeout in base_agent.py
    # as the safety net for a hung Gemini call. Revisit once confirmed.

    declarations = [
        types.FunctionDeclaration(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters_json_schema=tool.get("input_schema", {"type": "object", "properties": {}}),
        )
        for tool in tools
    ]
    gemini_tools = [types.Tool(function_declarations=declarations)] if declarations else None

    contents = [_to_gemini_content(m) for m in messages]

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


def _to_gemini_content(message: Any):
    """
    Messages in our history are either a plain {"role", "content": str}
    dict (simple text turns) or an already-native google.genai Content
    object (assistant/tool-result turns built by to_assistant_turn /
    to_tool_result_turn). Pass native objects through; convert dicts.
    """
    from google.genai import types

    if isinstance(message, types.Content):
        return message
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return types.Content(
            role=_to_gemini_role(message["role"]),
            parts=[types.Part.from_text(text=message["content"])],
        )
    raise TypeError(f"Unsupported message shape for Gemini: {message!r}")


def _to_gemini_role(role: str) -> str:
    """Gemini uses 'model' where Anthropic/OpenAI use 'assistant'."""
    return "model" if role == "assistant" else "user"
