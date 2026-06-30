"""
Ambient config context for tool execution.

The bug this fixes: tool functions (filesystem_tools.py, exec_tools.py,
monitoring_tools.py) were calling get_config() directly — which reads
fresh from environment variables / .env every time. That's correct for
the common case (one process, one .env, no isolation needed), but silently
wrong the moment an agent runs with a *different* Config than the global
one — which is exactly what eval_runner.py does to isolate each golden
task's sandbox/traces/state. The agent itself correctly used that custom
Config throughout (step limits, timeouts, tracing all go through
self.config) — but every tool it called ignored it and read the global
env-based sandbox/trace dir instead. Caught live: an eval task's agent
fixed a bug and ran tests successfully — against the shared ./workspace,
not the isolated eval sandbox the verification step actually checked.

base_agent.py sets this contextvar to self.config for the duration of each
tool call; tool functions read get_active_config() instead of calling
get_config() directly. Falls back to get_config() when nothing is set, so
calling a tool function standalone (e.g. directly in a test, exactly how
the existing filesystem/exec tool tests already do via monkeypatched env
vars) behaves exactly as before — this is additive, not a breaking change
to how tools can be invoked outside an agent run.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

from .config import Config, get_config

_current_config: contextvars.ContextVar[Config | None] = contextvars.ContextVar(
    "_current_config", default=None
)


def get_active_config() -> Config:
    """The config tool functions should use: whatever the calling agent set
    via use_config(), or the global env-based config if nothing is active."""
    cfg = _current_config.get()
    return cfg if cfg is not None else get_config()


@contextlib.contextmanager
def use_config(cfg: Config) -> Iterator[None]:
    """Makes `cfg` the active config for any tool calls made within this
    block. Always paired with a reset in `finally`, so an exception inside
    the block can't leave a stale config active for whatever runs next."""
    token = _current_config.set(cfg)
    try:
        yield
    finally:
        _current_config.reset(token)
