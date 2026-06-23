"""
Central configuration for the orchestrator.

Loads settings from environment variables (via .env in local dev, real env vars
in CI/production) and exposes them as one typed Config object. Nothing else in
the codebase should call os.environ directly — go through here, so there's a
single place that knows what configuration exists and fails fast if it's wrong.
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
    gemini_model: str
    anthropic_model: str
    max_steps_per_agent: int
    db_path: str


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
        max_steps_per_agent=int(os.getenv("MAX_STEPS_PER_AGENT", "15")),
        db_path=os.getenv("DB_PATH", "orchestrator_state.db"),
    )

    if provider == "gemini" and not cfg.gemini_api_key:
        raise RuntimeError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")
    if provider == "anthropic" and not cfg.anthropic_api_key:
        raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
    if provider not in ("gemini", "anthropic"):
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")

    return cfg
