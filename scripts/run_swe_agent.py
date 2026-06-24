"""
Manual smoke test for the SWE agent — makes real API calls, so it's not
part of the automated test suite (those are all mocked; see tests/unit).

Usage (after `conda activate agentops` and filling in .env):

    PYTHONPATH=src python scripts/run_swe_agent.py "create hello.txt containing 'hi'"

Importing filesystem_tools / exec_tools registers them into the shared
tool registry as a side effect of the import — that's why they're imported
here even though nothing else in this file calls them directly.
"""

from __future__ import annotations

import sys

from orchestrator.agents.swe_agent import SweAgent
from orchestrator.tools import exec_tools, filesystem_tools  # noqa: F401  (registers tools)


def main() -> None:
    task = sys.argv[1] if len(sys.argv) > 1 else "List the files in the sandbox directory."
    result = SweAgent().run(task)

    print(f"task_id:        {result.task_id}")
    print(f"stopped_reason: {result.stopped_reason}")
    print(f"steps_taken:    {result.steps_taken}")
    print(f"tokens:         in={result.total_input_tokens} out={result.total_output_tokens}")
    print(f"trace file:     traces/{result.task_id}.jsonl")
    print("--- final_text ---")
    print(result.final_text)


if __name__ == "__main__":
    main()
