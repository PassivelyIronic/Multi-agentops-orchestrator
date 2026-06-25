"""
Manual smoke test for a single agent role, run directly without going
through the orchestrator — useful for testing one role in isolation.
Makes real API calls, so it's not part of the automated test suite.

Usage:
    PYTHONPATH=src python scripts/run_agent.py swe "create hello.txt containing 'hi'"
    PYTHONPATH=src python scripts/run_agent.py tester "write tests for fizzbuzz.py"
    PYTHONPATH=src python scripts/run_agent.py oncall "check recent traces for any errors"
    PYTHONPATH=src python scripts/run_agent.py pm "we need user authentication"

The --agent choices are read from AVAILABLE_AGENTS, so this script doesn't
need updating when a new agent role is added in the future.
"""

from __future__ import annotations

import argparse

from orchestrator.orchestrator import AVAILABLE_AGENTS
from orchestrator.tools import (  # noqa: F401  (imports register tools as a side effect)
    exec_tools,
    filesystem_tools,
    monitoring_tools,
    search_tools,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", choices=sorted(AVAILABLE_AGENTS.keys()))
    parser.add_argument("task")
    args = parser.parse_args()

    agent_cls = AVAILABLE_AGENTS[args.agent]
    result = agent_cls().run(args.task)

    print(f"task_id:        {result.task_id}")
    print(f"stopped_reason: {result.stopped_reason}")
    if result.error:
        print(f"error:          {result.error}")
    print(f"steps_taken:    {result.steps_taken}")
    print(f"tokens:         in={result.total_input_tokens} out={result.total_output_tokens}")
    print("--- final_text ---")
    print(result.final_text)


if __name__ == "__main__":
    main()
