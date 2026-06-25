"""
Manual smoke test for a single agent role, run directly without going
through the orchestrator — useful for testing one role in isolation.
Makes real API calls, so it's not part of the automated test suite.

Usage:
    python scripts/run_agent.py swe "create hello.txt containing 'hi'"
    python scripts/run_agent.py tester "write tests for fizzbuzz.py"
    python scripts/run_agent.py oncall "check recent traces for any errors"
    python scripts/run_agent.py pm "we need user authentication"

No need to set PYTHONPATH first — see the sys.path note below.

The --agent choices are read from AVAILABLE_AGENTS, so this script doesn't
need updating when a new agent role is added in the future.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Makes `python scripts/run_agent.py` work directly, with no PYTHONPATH
# setup — `PYTHONPATH=src python ...` is bash syntax and doesn't work in
# cmd.exe (needs a separate `set` or `&&`-chaining there instead).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orchestrator.orchestrator import AVAILABLE_AGENTS  # noqa: E402
from orchestrator.tools import (  # noqa: E402,F401  (imports register tools as a side effect)
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
