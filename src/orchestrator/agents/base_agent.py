"""
Generic agent loop: call the model, execute any requested tools, feed
results back, repeat — until the model stops requesting tools, or one of
four limits is hit: step count, task wall-clock time, token budget, or a
repeated-call loop.

Concrete agents (SweAgent, ...) subclass this and only provide a system
prompt and the list of tool names they're allowed to use; everything below
is shared exactly once.

Every tool call goes through two checks before the real tool function ever
runs: (1) is this tool name in the *agent's own* tool_names — not just
"does it exist in the shared registry" — and (2) does it pass the
guardrails policy (guardrails.py). Either failing produces an error
ToolResult without ever touching the filesystem or a subprocess, and gets
recorded in the trace (tracing.py) exactly like a normal call. A tool that
raises is caught the same way — a bad tool call should let the model adapt,
not crash the whole task.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .. import guardrails
from ..config import Config, get_config
from ..llm_client import (
    ToolCall,
    ToolResult,
    call_with_tools,
    to_assistant_turn,
    to_tool_result_turn,
)
from ..tools.registry import ToolRegistry
from ..tools.registry import registry as default_registry
from ..tracing import TraceLogger, new_task_id


@dataclass
class AgentResult:
    final_text: str | None
    steps_taken: int
    # "done" | "step_limit" | "task_timeout" | "token_budget" | "repetition"
    stopped_reason: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    task_id: str = ""


class BaseAgent:
    system_prompt: str = ""
    # Subclasses override this with their own list — never mutated in place,
    # so sharing this empty list as a class-level default is safe.
    tool_names: list[str] = []

    def __init__(
        self, config: Config | None = None, tool_registry: ToolRegistry | None = None
    ) -> None:
        self.config = config or get_config()
        # Defaulting to the shared registry keeps normal usage simple
        # (`SweAgent().run(...)`); tests inject an isolated registry instead.
        self.registry = tool_registry or default_registry
        self.tracer: TraceLogger | None = None  # set fresh at the start of each run()

    def run(self, task: str, task_id: str | None = None) -> AgentResult:
        task_id = task_id or new_task_id()
        self.tracer = TraceLogger(task_id=task_id, config=self.config)

        messages: list = [{"role": "user", "content": task}]
        tools = self.registry.schemas_for(self.tool_names)

        total_in = total_out = 0
        last_signature: tuple | None = None
        repeat_count = 0
        start = time.monotonic()

        for step in range(1, self.config.max_steps_per_agent + 1):
            if time.monotonic() - start > self.config.task_timeout_seconds:
                return self._finish(None, step - 1, "task_timeout", total_in, total_out, task_id)

            call_start = time.monotonic()
            response = call_with_tools(
                messages, tools, system=self.system_prompt, config=self.config
            )
            self.tracer.log_llm_call(
                step, time.monotonic() - call_start, response.input_tokens, response.output_tokens
            )
            total_in += response.input_tokens
            total_out += response.output_tokens

            if total_in + total_out > self.config.max_tokens_per_task:
                return self._finish(
                    response.text, step, "token_budget", total_in, total_out, task_id
                )

            if not response.tool_calls:
                return self._finish(response.text, step, "done", total_in, total_out, task_id)

            signature = _signature_for(response.tool_calls)
            if signature == last_signature:
                repeat_count += 1
                if repeat_count >= self.config.repetition_limit:
                    return self._finish(
                        response.text, step, "repetition", total_in, total_out, task_id
                    )
            else:
                repeat_count = 0
            last_signature = signature

            results = [self._execute_tool(call, step) for call in response.tool_calls]

            messages.append(to_assistant_turn(response, self.config))
            messages.append(to_tool_result_turn(results, self.config))

        return self._finish(
            None, self.config.max_steps_per_agent, "step_limit", total_in, total_out, task_id
        )

    def _finish(
        self,
        final_text: str | None,
        steps_taken: int,
        stopped_reason: str,
        total_in: int,
        total_out: int,
        task_id: str,
    ) -> AgentResult:
        self.tracer.log_task_end(stopped_reason, steps_taken, total_in, total_out)
        return AgentResult(final_text, steps_taken, stopped_reason, total_in, total_out, task_id)

    def _execute_tool(self, call: ToolCall, step: int) -> ToolResult:
        tool_start = time.monotonic()

        if call.name not in self.tool_names:
            result = ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output=f"Tool '{call.name}' is not available to this agent.",
                is_error=True,
            )
            self._log_tool(step, call, result, blocked=True, latency=time.monotonic() - tool_start)
            return result

        violation = guardrails.check(call.name, call.arguments, self.config)
        if violation is not None:
            result = ToolResult(
                tool_call_id=call.id, name=call.name, output=violation.reason, is_error=True
            )
            self._log_tool(step, call, result, blocked=True, latency=time.monotonic() - tool_start)
            return result

        try:
            output = self.registry.execute(call.name, call.arguments)
            result = ToolResult(tool_call_id=call.id, name=call.name, output=output, is_error=False)
        except Exception as exc:
            result = ToolResult(
                tool_call_id=call.id, name=call.name, output=str(exc), is_error=True
            )

        self._log_tool(step, call, result, blocked=False, latency=time.monotonic() - tool_start)
        return result

    def _log_tool(
        self, step: int, call: ToolCall, result: ToolResult, blocked: bool, latency: float
    ) -> None:
        self.tracer.log_tool_call(
            step=step,
            name=call.name,
            arguments=call.arguments,
            output=result.output,
            is_error=result.is_error,
            blocked=blocked,
            latency_seconds=latency,
        )


def _signature_for(tool_calls: list[ToolCall]) -> tuple:
    """Order-independent fingerprint of a set of tool calls, for repetition detection."""
    return tuple(sorted((c.name, tuple(sorted(c.arguments.items()))) for c in tool_calls))
