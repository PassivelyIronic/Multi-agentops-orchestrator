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
| 4 | Remaining agents (Tester, On-call, PM) | ✅ |
| 5 | Eval harness (golden dataset, LLM-as-judge) | ⬜ |
| 6 | Dashboard + polish | ⬜ |

## Try it

```bash
conda env create -f environment.yml
conda activate agentops
cp .env.example .env   # fill in at least one API key
pytest                  # 112 tests, all mocked — no API key needed to run these

# Real end-to-end smoke test, single agent (uses your API key):
PYTHONPATH=src python scripts/run_swe_agent.py "create hello.txt containing 'hi'"

# Any agent role in isolation (swe | tester | oncall | pm):
PYTHONPATH=src python scripts/run_agent.py tester "write tests for fizzbuzz.py"

# Orchestrator: decomposes the task, routes subtasks across roles, checkpoints to SQLite
PYTHONPATH=src python scripts/run_orchestrator.py "create fizzbuzz.py with tests, then run them"
```

## Agent roles

| Role | Tools | Can write | Notes |
|---|---|---|---|
| `swe` | read/write/list/run_command | anywhere in the sandbox | implementation work |
| `tester` | read/write/list/run_command | `test_*.py` / `*_test.py` only | writes & runs tests, never fixes code — that's `swe`'s job |
| `oncall` | query_traces, health_check, read/list | nothing | read-only; investigates the project's own trace logs |
| `pm` | web_search, write/read/list | `*.md` only | turns a requirement into a prioritized markdown backlog |

`tester` and `swe` share the same tool *names* — what makes them different
roles is enforced by a guardrail (write path restricted by `agent_name`),
not just a system-prompt suggestion a model could ignore. `oncall` has no
`write_file` at all: least privilege for a role that only ever observes.

The SWE agent reads/writes files and runs shell commands inside `./workspace`
(configurable via `SANDBOX_DIR`). Every run writes a JSONL trace to
`./traces/<task_id>.jsonl` — which is exactly what `oncall`'s `query_traces`
tool reads; there's no separate fake "production log" concept, it
investigates the same operational data Phase 2 already generates. The
orchestrator additionally checkpoints its plan to `orchestrator_state.db`
(SQLite) — re-running `run_orchestrator.py` with `--task-id <same id>` after
an interruption resumes instead of restarting from scratch.

## LLM provider

Set `LLM_PROVIDER` in `.env` to `gemini`, `anthropic`, or `openrouter`.

- **Gemini** free tier on `gemini-2.5-flash`: ~5 requests/minute, **20 requests/day**.
  An orchestrator run (decomposition + several agent steps) burns through
  that in a single test.
- **OpenRouter** free tier (`:free` models, default `openai/gpt-oss-120b:free`):
  **50 requests/day, 20/minute** on a free/unfunded account; **1,000/day**
  once you've purchased $10+ in credits (the *models* stay free either
  way — paying just raises the account-level ceiling). Source:
  [OpenRouter rate limits FAQ](https://openrouter.ai/docs/faq).

**Gotcha that affects our own retry logic:** OpenRouter counts a failed
request — including a 429 from upstream provider throttling — against your
daily quota the same as a successful one. Our `MAX_RETRIES` (default 4)
retries on 429s automatically, so one throttled call can cost up to 5 of
your daily 50 requests instead of 1. If you're on an unfunded OpenRouter
account, consider setting `MAX_RETRIES=1` or `2` in `.env` — better to fail
fast and try again later than burn through the day's budget retrying.

## Why two LLM providers

Development happens for free against a Gemini key; Anthropic is used selectively
(e.g. evaluation runs) without touching agent code, because `llm_client.py`
normalizes both providers into one response shape. See `src/orchestrator/llm_client.py`.
