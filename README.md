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
| 1 | Single agent + tools (agent loop, step limits) | ⬜ |
| 2 | Guardrails + tracing | ⬜ |
| 3 | Orchestrator (task decomposition, state, routing) | ⬜ |
| 4 | Remaining agents (Tester, On-call, PM) | ⬜ |
| 5 | Eval harness (golden dataset, LLM-as-judge) | ⬜ |
| 6 | Dashboard + polish | ⬜ |

## Setup

```bash
conda env create -f environment.yml
conda activate agentops
cp .env.example .env   # fill in at least one API key
pytest
```

## Why two LLM providers

Development happens for free against a Gemini key; Anthropic is used selectively
(e.g. evaluation runs) without touching agent code, because `llm_client.py`
normalizes both providers into one response shape. See `src/orchestrator/llm_client.py`.
