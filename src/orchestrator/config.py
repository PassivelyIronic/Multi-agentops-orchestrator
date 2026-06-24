"""
Central configuration for the orchestrator.

Loads settings from environment variables (via .env in local dev, real env vars
in CI/production) and exposes them as one typed Config object. Nothing else in
the codebase should call os.environ directly — go through here, so there's a
single place that knows what configuration exists and fails fast if it's wrong.

Phase 1 additions beyond provider selection: resilience settings (retries,
token/cost budget, task timeout) and sandboxing (where agents are allowed to
read/write/execute). See docs/architecture.md for the reasoning behind each.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # no-op if no .env file is present (e.g. in CI)


@dataclass(frozen=True)
class Config:
    llm_provider: str  # "gemini" | "anthropic"
    gemini_api_key: str | None
    anthropic_api_key: str | None

    gemini_model: str = "gemini-2.5-flash"
    anthropic_model: str = "claude-sonnet-4-6"

    # --- Resilience -----------------------------------------------------
    # Retries cover transient failures (HTTP 429 rate limits, 5xx server
    # errors). They do NOT help with "daily quota exhausted" — that needs a
    # provider switch or waiting, which is why max_retries is small rather
    # than infinite.
    max_retries: int = 4
    retry_base_delay_seconds: float = 1.0
    llm_request_timeout_seconds: int = 60

    # Token budget is tracked separately from step count: a single tool
    # result (e.g. a big file read) can blow the budget without using many
    # steps.
    max_tokens_per_task: int = 300_000
    max_steps_per_agent: int = 15
    task_timeout_seconds: int = 600
    repetition_limit: int = 3  # abort after N identical tool calls in a row

    # --- Sandboxing -------------------------------------------------------
    # Filesystem and exec tools are confined to this directory. Full
    # guardrails (allow/deny command lists, audit logging) land in Phase 2 —
    # this is the non-negotiable minimum until then, since these tools touch
    # the real filesystem the moment an agent runs.
    sandbox_dir: str = "./workspace"
    tool_exec_timeout_seconds: int = 30

    # --- Guardrails (Phase 2) ---------------------------------------------
    # Configurable policy layer, checked before a tool runs — separate from
    # the hardcoded sandboxing above. None means "use the built-in default
    # deny-list"; set via env as a comma-separated list of regex patterns to
    # override it entirely.
    denied_command_patterns: tuple[str, ...] | None = None
    max_write_bytes: int = 1_000_000

    # --- Tracing (Phase 2) -------------------------------------------------
    # One JSONL file per task under trace_dir. Cost fields default to 0 —
    # we don't hardcode provider prices since they go stale; fill in the
    # *_PRICE_PER_MILLION env vars from current provider pricing if you
    # want real $ figures instead of token counts alone.
    trace_dir: str = "./traces"
    gemini_input_price_per_million: float = 0.0
    gemini_output_price_per_million: float = 0.0
    anthropic_input_price_per_million: float = 0.0
    anthropic_output_price_per_million: float = 0.0

    db_path: str = "orchestrator_state.db"


def get_config() -> Config:
    """
    Build a Config from environment variables.

    Raises immediately if the selected provider is missing its API key —
    we want that failure at startup, not three steps into an agent run.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    cfg = Config(
        llm_provider=provider,
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_retries=int(os.getenv("MAX_RETRIES", "4")),
        retry_base_delay_seconds=float(os.getenv("RETRY_BASE_DELAY_SECONDS", "1.0")),
        llm_request_timeout_seconds=int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "60")),
        max_tokens_per_task=int(os.getenv("MAX_TOKENS_PER_TASK", "300000")),
        max_steps_per_agent=int(os.getenv("MAX_STEPS_PER_AGENT", "15")),
        task_timeout_seconds=int(os.getenv("TASK_TIMEOUT_SECONDS", "600")),
        repetition_limit=int(os.getenv("REPETITION_LIMIT", "3")),
        sandbox_dir=os.getenv("SANDBOX_DIR", "./workspace"),
        tool_exec_timeout_seconds=int(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "30")),
        denied_command_patterns=_parse_patterns(os.getenv("DENIED_COMMAND_PATTERNS")),
        max_write_bytes=int(os.getenv("MAX_WRITE_BYTES", "1000000")),
        trace_dir=os.getenv("TRACE_DIR", "./traces"),
        gemini_input_price_per_million=float(os.getenv("GEMINI_INPUT_PRICE_PER_MILLION", "0.0")),
        gemini_output_price_per_million=float(os.getenv("GEMINI_OUTPUT_PRICE_PER_MILLION", "0.0")),
        anthropic_input_price_per_million=float(
            os.getenv("ANTHROPIC_INPUT_PRICE_PER_MILLION", "0.0")
        ),
        anthropic_output_price_per_million=float(
            os.getenv("ANTHROPIC_OUTPUT_PRICE_PER_MILLION", "0.0")
        ),
        db_path=os.getenv("DB_PATH", "orchestrator_state.db"),
    )

    if provider == "gemini" and not cfg.gemini_api_key:
        raise RuntimeError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")
    if provider == "anthropic" and not cfg.anthropic_api_key:
        raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
    if provider not in ("gemini", "anthropic"):
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")

    return cfg


def _parse_patterns(raw: str | None) -> tuple[str, ...] | None:
    """Comma-separated env var -> tuple of patterns, or None if unset (use defaults)."""
    if not raw:
        return None
    return tuple(p.strip() for p in raw.split(",") if p.strip())
