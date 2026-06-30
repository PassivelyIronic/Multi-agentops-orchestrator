"""
Manual smoke test for the SWE agent — makes real API calls, so it's not
part of the automated test suite (those are all mocked; see tests/unit).

Usage (after `conda activate agentops` and filling in .env):

    python scripts/run_swe_agent.py "create hello.txt containing 'hi'"

No need to set PYTHONPATH first — see the sys.path note below.

Importing filesystem_tools / exec_tools registers them into the shared
tool registry as a side effect of the import — that's why they're imported
here even though nothing else in this file calls them directly.

The task_id is resolved and printed *before* run() is called — if a
quota/rate-limit error exhausts all retries, run() returns a normal
"api_error" result rather than raising, but printing the id upfront means
you always have it even if something unexpected still goes wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Makes `python scripts/run_swe_agent.py` work directly, with no PYTHONPATH
# setup — needed because `src` layout means `orchestrator` isn't importable
# otherwise, and `PYTHONPATH=src python ...` is bash syntax that doesn't
# work in cmd.exe (the inline-env-var-then-command form is bash-specific;
# cmd.exe needs a separate `set` command or `&&`-chaining instead).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orchestrator.agents.swe_agent import SweAgent  # noqa: E402
from orchestrator.tools import exec_tools, filesystem_tools  # noqa: E402,F401  (registers tools)
from orchestrator.tracing import new_task_id  # noqa: E402


def main() -> None:
    task = sys.argv[1] if len(sys.argv) > 1 else "List the files in the sandbox directory."
    task_id = new_task_id()
    print(f"task_id:        {task_id}")

    result = SweAgent().run(task, task_id=task_id)

    print(f"stopped_reason: {result.stopped_reason}")
    if result.error:
        print(f"error:          {result.error}")
    print(f"steps_taken:    {result.steps_taken}")
    print(f"tokens:         in={result.total_input_tokens} out={result.total_output_tokens}")
    print(f"trace file:     traces/{result.task_id}.jsonl")
    print("--- final_text ---")
    print(result.final_text)


if __name__ == "__main__":
    main()
