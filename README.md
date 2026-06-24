# AgentOps Orchestrator

A multi-agent system that simulates a small dev team — Software Engineer, Tester,
On-call, and Product Manager agents — coordinated by an orchestrator with tool use,
guardrails, and an evaluation harness.

<!-- Replace OWNER below once pushed to GitHub -->
![CI](https://github.com/OWNER/agentops-orchestrator/actions/workflows/ci.yml/badge.svg)

## What this does

Given a task (e.g. "add a `/health` endpoint", "fix a failing test"), the orchestrator
decomposes it and routes work to specialized agents. Each agent uses a defined set of
tools (filesystem, exec, git, search, monitoring) under step limits and guardrails.
Every tool call is traced — tokens, cost, latency — and viewable in a live dashboard.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the component diagram and
rationale (including why there's a provider-agnostic LLM client behind two different
model APIs).

## Status

🚧 Work in progress.

| Phase | What | Status |
|---|---|---|
| 0 | Project skeleton, CI, config | ✅ |
| 1 | Single agent + tools (agent loop, step limits, resilience, sandboxing) | ✅ |
| 2 | Guardrails (command/path policy, audit-logged blocks) + structured tracing | ✅ |
| 3 | Orchestrator (task decomposition, SQLite state, routing, resumability) | ✅ |
| 4 | Remaining agents (Tester, On-call, PM) | ⬜ |
| 5 | Eval harness (golden dataset, LLM-as-judge) | ⬜ |
| 6 | Dashboard + polish | ⬜ |

## Try it

```bash
conda env create -f environment.yml
conda activate agentops
cp .env.example .env   # fill in at least one API key
pytest                  # 69 tests, all mocked — no API key needed to run these

# Real end-to-end smoke test, single agent (uses your API key):
PYTHONPATH=src python scripts/run_swe_agent.py "create hello.txt containing 'hi'"

# Orchestrator: decomposes the task, routes subtasks, checkpoints to SQLite
PYTHONPATH=src python scripts/run_orchestrator.py "create fizzbuzz.py with tests, then run them"
```

The SWE agent reads/writes files and runs shell commands inside `./workspace`
(configurable via `SANDBOX_DIR`). Every run writes a JSONL trace to
`./traces/<task_id>.jsonl`. The orchestrator additionally checkpoints its
plan to `orchestrator_state.db` (SQLite) — re-running `run_orchestrator.py`
with `--task-id <same id>` after an interruption resumes instead of
restarting from scratch.

## Why two LLM providers

Development happens for free against a Gemini key; Anthropic is used selectively
(e.g. evaluation runs) without touching agent code, because `llm_client.py`
normalizes both providers into one response shape. See `src/orchestrator/llm_client.py`.
