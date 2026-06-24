"""
Task state persistence (SQLite).

Stores the orchestrator's plan (subtasks) and progress (status per
subtask) so a crashed or restarted process can resume a multi-step task
instead of starting over — this is what turns "loop that runs until done"
into "long-running task that survives a restart".

Two tables: `tasks` (one row per top-level task) and `subtasks` (one row
per planned subtask, in order). A subtask's status moves
pending -> running -> done|failed; resuming a task re-runs any subtask
still in pending OR running — a `running` subtask that never finished
(process was killed mid-step) is retried, never assumed done.

Each function opens its own short-lived connection rather than holding one
open — this mirrors how a real restarted process would reopen the db, and
is what the tests actually exercise (state surviving across separate
connections, not just within one Python process).
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .config import Config, get_config

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    original_task TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS subtasks (
    task_id TEXT NOT NULL,
    subtask_index INTEGER NOT NULL,
    agent TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    PRIMARY KEY (task_id, subtask_index)
);
"""


@dataclass
class SubtaskRecord:
    subtask_index: int
    agent: str
    description: str
    status: str  # "pending" | "running" | "done" | "failed"
    result_json: str | None = None


@contextmanager
def _connect(config: Config):
    conn = sqlite3.connect(config.db_path)
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_plan(
    task_id: str, original_task: str, subtasks: Iterable[Any], config: Config | None = None
) -> None:
    """Persist a freshly-decomposed plan. Subtasks start as 'pending'.

    `subtasks` items just need `.agent` and `.description` attributes —
    typed as Subtask in orchestrator.py, kept untyped here to avoid a
    circular import (orchestrator.py imports this module, not vice versa).
    """
    cfg = config or get_config()
    with _connect(cfg) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO tasks (task_id, original_task, status, created_at) "
            "VALUES (?, ?, 'running', ?)",
            (task_id, original_task, time.time()),
        )
        for i, sub in enumerate(subtasks):
            conn.execute(
                "INSERT OR REPLACE INTO subtasks "
                "(task_id, subtask_index, agent, description, status, result_json) "
                "VALUES (?, ?, ?, ?, 'pending', NULL)",
                (task_id, i, sub.agent, sub.description),
            )


def load_subtasks(task_id: str, config: Config | None = None) -> list[SubtaskRecord]:
    cfg = config or get_config()
    with _connect(cfg) as conn:
        rows = conn.execute(
            "SELECT subtask_index, agent, description, status, result_json "
            "FROM subtasks WHERE task_id = ? ORDER BY subtask_index",
            (task_id,),
        ).fetchall()
    return [SubtaskRecord(*row) for row in rows]


def mark_subtask_running(task_id: str, index: int, config: Config | None = None) -> None:
    cfg = config or get_config()
    with _connect(cfg) as conn:
        conn.execute(
            "UPDATE subtasks SET status = 'running' WHERE task_id = ? AND subtask_index = ?",
            (task_id, index),
        )


def mark_subtask_done(
    task_id: str, index: int, result_summary: str, config: Config | None = None
) -> None:
    cfg = config or get_config()
    with _connect(cfg) as conn:
        conn.execute(
            "UPDATE subtasks SET status = 'done', result_json = ? "
            "WHERE task_id = ? AND subtask_index = ?",
            (json.dumps(result_summary), task_id, index),
        )


def mark_subtask_failed(task_id: str, index: int, error: str, config: Config | None = None) -> None:
    cfg = config or get_config()
    with _connect(cfg) as conn:
        conn.execute(
            "UPDATE subtasks SET status = 'failed', result_json = ? "
            "WHERE task_id = ? AND subtask_index = ?",
            (json.dumps({"error": error}), task_id, index),
        )


def mark_task_status(task_id: str, status: str, config: Config | None = None) -> None:
    cfg = config or get_config()
    with _connect(cfg) as conn:
        conn.execute("UPDATE tasks SET status = ? WHERE task_id = ?", (status, task_id))
