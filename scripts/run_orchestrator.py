"""
Manual smoke test for the orchestrator — makes real API calls (one for
decomposition, more for each routed subtask), so it's not part of the
automated test suite (those mock everything; see tests/unit/test_orchestrator.py).

Usage:
    PYTHONPATH=src python scripts/run_orchestrator.py "create fizzbuzz.py with tests, then run them"

To see resumability: note the printed task_id, kill the process (Ctrl+C)
after a subtask finishes, then re-run with --task-id <same id>. It will
skip straight to the remaining subtask instead of re-decomposing or
re-running the one already marked done — that's the whole point of
state.py existing.
"""

from __future__ import annotations

import argparse

from orchestrator.orchestrator import run
from orchestrator.tools import exec_tools, filesystem_tools  # noqa: F401  (registers tools)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument(
        "--task-id", default=None, help="Reuse a task_id to resume an interrupted run"
    )
    args = parser.parse_args()

    result = run(args.task, task_id=args.task_id)

    print(f"task_id:        {result.task_id}")
    print(f"stopped_reason: {result.stopped_reason}")
    for subtask, agent_result in result.subtask_results:
        print(f"\n[{subtask.agent}] {subtask.description}")
        print(f"  stopped_reason: {agent_result.stopped_reason}")
        print(f"  final_text: {agent_result.final_text}")


if __name__ == "__main__":
    main()
