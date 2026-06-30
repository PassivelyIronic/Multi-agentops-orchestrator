"""
Golden dataset loading, task execution, and outcome verification for the
eval harness.

Each golden task runs in its own isolated sandbox/trace/state directory
under a temp root — never the shared ./workspace — so eval runs are
reproducible and don't get contaminated by leftover files from previous
interactive testing. This isolation exists *because* of a real failure
mode hit during manual Phase 4 testing: a leftover test_fizzbuzz.py from
an earlier demo run confused a later one. Eval results need to be trusted
more than that.

Verification is mechanical wherever possible (file existence/content,
command exit code) and falls back to eval_judge.judge() only for genuinely
qualitative criteria — mechanical checks are deterministic and free;
LLM-as-judge costs a call and is a probabilistic approximation, so it's
the fallback, not the default.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .config import Config, get_config
from .eval_judge import judge
from .orchestrator import AVAILABLE_AGENTS
from .orchestrator import run as run_orchestrator

_DEFAULT_COMMAND_TIMEOUT_SECONDS = 30


@dataclass
class GoldenTask:
    id: str
    agent: str  # one of AVAILABLE_AGENTS' keys, or "orchestrator"
    task: str
    verification: Any  # a single spec dict, or a list of spec dicts (all must pass)
    setup_sandbox: dict[str, str] = field(default_factory=dict)  # relative path -> content
    setup_traces: dict[str, str] = field(default_factory=dict)  # filename -> raw JSONL content


@dataclass
class TaskEvalResult:
    task_id: str
    passed: bool
    verification_detail: str
    stopped_reason: str
    steps_taken: int
    total_input_tokens: int
    total_output_tokens: int
    wall_seconds: float
    error: str | None = None


@dataclass
class EvalReport:
    results: list[TaskEvalResult]

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.pass_count / len(self.results) if self.results else 0.0

    @property
    def total_input_tokens(self) -> int:
        return sum(r.total_input_tokens for r in self.results)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.total_output_tokens for r in self.results)

    def to_markdown(self) -> str:
        lines = [
            "# Eval Report",
            "",
            f"**Pass rate:** {self.pass_rate:.0%} ({self.pass_count}/{len(self.results)})",
            f"**Total tokens:** in={self.total_input_tokens} out={self.total_output_tokens}",
            "",
            "| Task | Result | Stopped reason | Steps | Tokens (in/out) | Time (s) |",
            "|---|---|---|---|---|---|",
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(
                f"| {r.task_id} | {status} | {r.stopped_reason} | {r.steps_taken} | "
                f"{r.total_input_tokens}/{r.total_output_tokens} | {r.wall_seconds:.1f} |"
            )

        failures = [r for r in self.results if not r.passed]
        if failures:
            lines += ["", "## Failure details", ""]
            for r in failures:
                lines.append(f"### {r.task_id}")
                lines.append("")
                lines.append(f"```\n{r.verification_detail}\n```")
                if r.error:
                    lines.append(f"\nAgent-level error: `{r.error}`")
                lines.append("")
        return "\n".join(lines)


def load_golden_dataset(path: str | Path) -> list[GoldenTask]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            tasks.append(
                GoldenTask(
                    id=data["id"],
                    agent=data["agent"],
                    task=data["task"],
                    verification=data["verification"],
                    setup_sandbox=data.get("setup_sandbox", {}),
                    setup_traces=data.get("setup_traces", {}),
                )
            )
    return tasks


def run_golden_dataset(
    dataset_path: str | Path, work_root: Path, config: Config | None = None
) -> EvalReport:
    tasks = load_golden_dataset(dataset_path)
    results = [run_task(t, work_root, config) for t in tasks]
    return EvalReport(results)


def run_task(
    golden_task: GoldenTask, work_root: Path, base_config: Config | None = None
) -> TaskEvalResult:
    cfg_base = base_config or get_config()
    task_dir = work_root / golden_task.id
    sandbox_dir = task_dir / "workspace"
    trace_dir = task_dir / "traces"

    # Wipe any leftover state from a previous run of this exact task_id —
    # setup_sandbox below only overwrites the files it explicitly lists, so
    # without this, anything an agent wrote beyond that (extra files,
    # __pycache__, a stale state.db) would carry over and make repeated
    # runs of the same task non-reproducible.
    shutil.rmtree(task_dir, ignore_errors=True)
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    for rel_path, content in golden_task.setup_sandbox.items():
        target = sandbox_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    for filename, content in golden_task.setup_traces.items():
        (trace_dir / filename).write_text(content, encoding="utf-8")

    # Isolates this task's filesystem/state footprint while keeping every
    # other setting (provider, model, retries, ...) from the base config.
    cfg = replace(
        cfg_base,
        sandbox_dir=str(sandbox_dir),
        trace_dir=str(trace_dir),
        db_path=str(task_dir / "state.db"),
    )

    start = time.monotonic()
    error: str | None = None
    stopped_reason = "unknown"
    steps_taken = 0
    total_in = total_out = 0
    final_text = ""

    try:
        if golden_task.agent == "orchestrator":
            result = run_orchestrator(golden_task.task, config=cfg)
            stopped_reason = result.stopped_reason
            steps_taken = sum(r.steps_taken for _, r in result.subtask_results)
            total_in = sum(r.total_input_tokens for _, r in result.subtask_results)
            total_out = sum(r.total_output_tokens for _, r in result.subtask_results)
            if result.subtask_results:
                final_text = result.subtask_results[-1][1].final_text or ""
        elif golden_task.agent in AVAILABLE_AGENTS:
            agent_cls = AVAILABLE_AGENTS[golden_task.agent]
            agent_result = agent_cls(config=cfg).run(golden_task.task)
            stopped_reason = agent_result.stopped_reason
            steps_taken = agent_result.steps_taken
            total_in = agent_result.total_input_tokens
            total_out = agent_result.total_output_tokens
            final_text = agent_result.final_text or ""
        else:
            error = f"Unknown agent in golden task: {golden_task.agent!r}"
    except Exception as exc:
        # Agents themselves shouldn't raise (see base_agent.py's api_error
        # handling) — this is a safety net for the eval harness itself,
        # so one broken task can't take down the whole dataset run.
        error = str(exc)

    wall_seconds = time.monotonic() - start
    passed, detail = _verify_all(golden_task.verification, sandbox_dir, final_text, cfg)

    return TaskEvalResult(
        task_id=golden_task.id,
        passed=passed and error is None,
        verification_detail=detail if error is None else f"{detail}\nAgent error: {error}",
        stopped_reason=stopped_reason,
        steps_taken=steps_taken,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        wall_seconds=wall_seconds,
        error=error,
    )


def _verify_all(
    verification_spec: Any, sandbox_dir: Path, final_text: str, cfg: Config
) -> tuple[bool, str]:
    specs = verification_spec if isinstance(verification_spec, list) else [verification_spec]
    details = []
    all_passed = True
    for spec in specs:
        passed, detail = _verify_one(spec, sandbox_dir, final_text, cfg)
        details.append(detail)
        all_passed = all_passed and passed
    return all_passed, "\n".join(details)


def _verify_one(
    spec: dict[str, Any], sandbox_dir: Path, final_text: str, cfg: Config
) -> tuple[bool, str]:
    vtype = spec.get("type")

    if vtype == "file_exists":
        path = sandbox_dir / spec["path"]
        exists = path.is_file()
        return exists, f"file_exists({spec['path']!r}) -> {exists}"

    if vtype == "file_contains":
        path = sandbox_dir / spec["path"]
        if not path.is_file():
            return False, f"file_contains: {spec['path']!r} does not exist"
        content = path.read_text(encoding="utf-8")
        found = spec["substring"] in content
        return found, f"file_contains({spec['path']!r}, {spec['substring']!r}) -> {found}"

    if vtype == "file_unchanged":
        path = sandbox_dir / spec["path"]
        if not path.is_file():
            return False, f"file_unchanged: {spec['path']!r} does not exist"
        matches = path.read_text(encoding="utf-8") == spec["expected_content"]
        return matches, f"file_unchanged({spec['path']!r}) -> {matches}"

    if vtype == "no_files_matching":
        matches = list(sandbox_dir.glob(spec["pattern"]))
        passed = len(matches) == 0
        return (
            passed,
            f"no_files_matching({spec['pattern']!r}) -> found {[m.name for m in matches]}",
        )

    if vtype == "command_succeeds":
        try:
            result = subprocess.run(
                shlex.split(spec["command"]),
                cwd=sandbox_dir,
                capture_output=True,
                text=True,
                timeout=spec.get("timeout_seconds", _DEFAULT_COMMAND_TIMEOUT_SECONDS),
            )
        except Exception as exc:
            return False, f"command_succeeds({spec['command']!r}): failed to run ({exc})"
        passed = result.returncode == 0
        detail = f"command_succeeds({spec['command']!r}) -> exit_code={result.returncode}"
        if not passed:
            detail += f"\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
        return passed, detail

    if vtype == "llm_judge":
        if "context_file" in spec:
            context_path = sandbox_dir / spec["context_file"]
            context = context_path.read_text(encoding="utf-8") if context_path.is_file() else ""
        else:
            context = final_text
        passed, reasoning = judge(spec["rubric"], context, cfg)
        return passed, f"llm_judge -> {passed}: {reasoning}"

    return False, f"Unknown verification type: {vtype!r}"
