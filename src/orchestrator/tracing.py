"""
Structured trace logging: every LLM call and every tool execution within a
task gets one JSON line, with enough detail (tokens, latency, whether a
guardrail blocked it, whether it errored) to reconstruct what happened
without re-running anything. This is what Phase 6's dashboard reads, and
what Phase 5's eval harness pulls cost/latency numbers from.

One file per task: traces/<task_id>.jsonl — newline-delimited JSON, trivially
appendable and streamable, no database needed yet. Phase 3's state.py covers
task *state* (current step, resumability); this is a separate, append-only
record of what happened, which is a different access pattern and doesn't
need to be queried mid-task the way state does.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, get_config

_MAX_LOGGED_CHARS = 2_000  # truncate large tool I/O before it hits the trace file


def new_task_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class TraceLogger:
    task_id: str
    config: Config = field(default_factory=get_config)

    def __post_init__(self) -> None:
        self._path = Path(self.config.trace_dir) / f"{self.task_id}.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_llm_call(
        self, step: int, latency_seconds: float, input_tokens: int, output_tokens: int
    ) -> None:
        self._write(
            {
                "type": "llm_call",
                "step": step,
                "latency_seconds": round(latency_seconds, 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": _estimate_cost(self.config, input_tokens, output_tokens),
            }
        )

    def log_tool_call(
        self,
        step: int,
        name: str,
        arguments: dict[str, Any],
        output: str,
        is_error: bool,
        blocked: bool,
        latency_seconds: float,
    ) -> None:
        self._write(
            {
                "type": "tool_call",
                "step": step,
                "name": name,
                "arguments": _truncate(json.dumps(arguments, default=str)),
                "output": _truncate(output),
                "is_error": is_error,
                "blocked": blocked,
                "latency_seconds": round(latency_seconds, 3),
            }
        )

    def log_task_end(
        self,
        stopped_reason: str,
        steps_taken: int,
        total_input_tokens: int,
        total_output_tokens: int,
    ) -> None:
        self._write(
            {
                "type": "task_end",
                "stopped_reason": stopped_reason,
                "steps_taken": steps_taken,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "estimated_cost_usd": _estimate_cost(
                    self.config, total_input_tokens, total_output_tokens
                ),
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        record["timestamp"] = time.time()
        record["task_id"] = self.task_id
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _truncate(text: str) -> str:
    if len(text) > _MAX_LOGGED_CHARS:
        return text[:_MAX_LOGGED_CHARS] + "...[truncated]"
    return text


def _estimate_cost(cfg: Config, input_tokens: int, output_tokens: int) -> float:
    """
    Returns 0.0 unless pricing has been configured via env vars. Provider
    prices aren't hardcoded here since they change and go stale — fill in
    *_INPUT_PRICE_PER_MILLION / *_OUTPUT_PRICE_PER_MILLION from current
    provider pricing pages if you want real $ figures instead of raw
    token counts alone.
    """
    if cfg.llm_provider == "gemini":
        in_price = cfg.gemini_input_price_per_million
        out_price = cfg.gemini_output_price_per_million
    else:
        in_price = cfg.anthropic_input_price_per_million
        out_price = cfg.anthropic_output_price_per_million
    return round(input_tokens / 1_000_000 * in_price + output_tokens / 1_000_000 * out_price, 6)
