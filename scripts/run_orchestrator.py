"""
Manual smoke test for the orchestrator — makes real API calls (one for
decomposition, more for each routed subtask), so it's not part of the
automated test suite (those mock everything; see tests/unit/test_orchestrator.py).

Usage:
    PYTHONPATH=src python scripts/run_orchestrator.py "create fizzbuzz.py with tests, then run them"

To see resumability: note the printed task_id, kill the process (Ctrl+C)
after a subtask finishes, then re-run with --task-id <same id>. It will
skip straight to the remaining subtask instead of re-decomposing or
re-running the one already marked done.

The task_id is resolved and printed *before* run() is called, not after —
if a quota/rate-limit error exhausts all retries, run() returns a normal
result rather than raising (see base_agent.py / orchestrator.py), but
printing the id upfront means you always have it even if something
unexpected still goes wrong.
"""

from __future__ import annotations

import argparse

from orchestrator.orchestrator import run
from orchestrator.tools import (  # noqa: F401  (imports register tools as a side effect)
    exec_tools,
    filesystem_tools,
    monitoring_tools,
    search_tools,
)
from orchestrator.tracing import new_task_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument(
        "--task-id", default=None, help="Reuse a task_id to resume an interrupted run"
    )
    args = parser.parse_args()

    task_id = args.task_id or new_task_id()
    print(f"task_id: {task_id}")

    result = run(args.task, task_id=task_id)

    print(f"stopped_reason: {result.stopped_reason}")
    if result.error:
        print(f"error: {result.error}")
    for subtask, agent_result in result.subtask_results:
        print(f"\n[{subtask.agent}] {subtask.description}")
        print(f"  stopped_reason: {agent_result.stopped_reason}")
        if agent_result.error:
            print(f"  error: {agent_result.error}")
        print(f"  final_text: {agent_result.final_text}")

    if result.stopped_reason != "done":
        print(f"\nNot finished — resume later with: --task-id {task_id}")


if __name__ == "__main__":
    main()
