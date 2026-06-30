# AgentOps Orchestrator

A multi-agent system that simulates a small dev team — Software Engineer, Tester,
On-call, and Product Manager agents — coordinated by an orchestrator with tool use,
guardrails, and an evaluation harness.


## What this does

Given a task (e.g. "add a `/health` endpoint", "fix a failing test"), the orchestrator
decomposes it and routes work to specialized agents. Each agent uses a defined set of
tools (filesystem, exec, web search, monitoring) under step limits and guardrails.
Every tool call is traced — tokens, cost, latency — and viewable in a live dashboard.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the component diagram and
rationale (including why there's a provider-agnostic LLM client behind three
different model APIs).

## Status

All 6 planned phases are implemented and tested (151 tests). `git_tools.py` is
the one deliberately-unimplemented stub left from Phase 1 — committing is
exactly the kind of action that should go through a guardrails review before
shipping it, and that review hasn't happened yet; see its docstring.

| Phase | What | Status |
|---|---|---|
| 0 | Project skeleton, CI, config | ✅ |
| 1 | Single agent + tools (agent loop, step limits, resilience, sandboxing) | ✅ |
| 2 | Guardrails (command/path policy, audit-logged blocks) + structured tracing | ✅ |
| 3 | Orchestrator (task decomposition, SQLite state, routing, resumability) | ✅ |
| 4 | Remaining agents (Tester, On-call, PM) | ✅ |
| 5 | Eval harness (golden dataset, LLM-as-judge) | ✅ |
| 6 | Dashboard + polish | ✅ |

## Try it

```bash
conda env create -f environment.yml
conda activate agentops
cp .env.example .env   # fill in at least one API key
pytest                  # 151 tests, all mocked — no API key needed to run these

# Real end-to-end smoke test, single agent (uses your API key):
python scripts/run_swe_agent.py "create hello.txt containing 'hi'"

# Any agent role in isolation (swe | tester | oncall | pm):
python scripts/run_agent.py tester "write tests for fizzbuzz.py"

# Orchestrator: decomposes the task, routes subtasks across roles, checkpoints to SQLite
python scripts/run_orchestrator.py "create fizzbuzz.py with tests, then run them"
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

## Eval harness

```bash
python eval/run_eval.py                              # full golden dataset (10 tasks)
python eval/run_eval.py --ids swe_fizzbuzz,swe_bugfix # a subset
```

Each task runs in its own isolated directory under `eval/runs/<task_id>/` —
never the shared `./workspace` — so results are reproducible. Verification
is mechanical wherever possible (file exists/content, command exit code)
and falls back to an LLM-as-judge call only for genuinely qualitative
criteria (e.g. "is this backlog well-prioritized?"). Two of the ten tasks
exist specifically to test guardrails, not just capability:
`tester_respects_scope` verifies the implementation file is byte-identical
after a task that tempts the tester into "helpfully" refactoring it, and
`pm_writes_markdown_only` verifies no `.py` file gets created. A markdown
report (pass rate, tokens, per-task detail, failure reasoning) is written
to `eval/report.md`.

Cost note: a full run is ~25-35 LLM calls — a meaningful chunk of
OpenRouter's free-tier daily budget (see above). Use `--ids` for a subset,
or a funded account / Gemini for a full run.

## Dashboard

```bash
streamlit run dashboard/app.py
```

Reads the exact same JSONL traces and SQLite state every other part of
this project already writes — no separate data pipeline. Shows: overall
success rate and token totals, a sortable table of recent tasks, a
per-task timeline (every LLM call, tool call, and error in order, with
latency/tokens/blocked flags), and orchestrator plans with their subtask
breakdown. Auto-refreshes every 5 seconds, so you can leave it open in a
second window while running tasks in a terminal.

