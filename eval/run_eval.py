"""
Eval harness CLI: runs the golden dataset against real agents/orchestrator
and produces a pass/fail report. Makes real API calls (one set per task,
roughly 1-10 LLM calls depending on the task) — this is not part of the
automated test suite (tests/unit mocks everything; see
tests/unit/test_eval_runner.py and test_eval_judge.py).

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --dataset eval/golden_dataset.jsonl --out eval/report.md
    python eval/run_eval.py --ids swe_fizzbuzz,swe_bugfix   # run a subset

Each task runs in its own isolated directory under eval/runs/<task_id>/ —
never the shared ./workspace — so results are reproducible and don't
inherit leftover files from interactive testing.

Cost note: running the full dataset is ~25-35 LLM calls total. On
OpenRouter's free tier (50 requests/day for an unfunded account — see
README's "LLM provider" section) that's a meaningful chunk of a day's
budget; consider --ids to run a subset, or Gemini/a funded account for a
full run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Makes `python eval/run_eval.py` work directly, with no PYTHONPATH setup.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orchestrator.eval_runner import (  # noqa: E402
    EvalReport,  # noqa: E402
    load_golden_dataset,
    run_task,
)
from orchestrator.tools import (  # noqa: E402,F401  (imports register tools as a side effect)
    exec_tools,
    filesystem_tools,
    monitoring_tools,
    search_tools,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(Path(__file__).parent / "golden_dataset.jsonl"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "report.md"))
    parser.add_argument(
        "--work-root",
        default=str(Path(__file__).parent / "runs"),
        help="Where each task's isolated sandbox/traces/state lives.",
    )
    parser.add_argument(
        "--ids", default=None, help="Comma-separated task ids to run (default: all)"
    )
    args = parser.parse_args()

    tasks = load_golden_dataset(args.dataset)
    if args.ids:
        wanted = set(args.ids.split(","))
        tasks = [t for t in tasks if t.id in wanted]
        missing = wanted - {t.id for t in tasks}
        if missing:
            print(f"Warning: unknown task id(s) ignored: {sorted(missing)}")

    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    results = []
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] running {task.id} ({task.agent})...", flush=True)
        result = run_task(task, work_root)
        status = "PASS" if result.passed else "FAIL"
        print(
            f"    -> {status}  ({result.stopped_reason}, {result.steps_taken} steps, "
            f"{result.wall_seconds:.1f}s)"
        )
        results.append(result)

    report = EvalReport(results)
    print(f"\nPass rate: {report.pass_rate:.0%} ({report.pass_count}/{len(results)})")
    print(f"Total tokens: in={report.total_input_tokens} out={report.total_output_tokens}")

    Path(args.out).write_text(report.to_markdown(), encoding="utf-8")
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
