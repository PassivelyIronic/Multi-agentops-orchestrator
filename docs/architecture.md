# Architecture

## Overview

```mermaid
flowchart TB
    U[Incoming task] --> O[Orchestrator]
    O --> SWE[Software Engineer Agent]
    O --> T[Tester Agent]
    O --> OC[On-call Agent]
    O --> PM[Product Manager Agent]
    SWE --> FS[filesystem / exec / git tools]
    T --> EX[exec tools]
    OC --> MON[monitoring tools]
    PM --> SR[search tools]
    SWE -. traced .-> TR[Tracing]
    T -. traced .-> TR
    OC -. traced .-> TR
    PM -. traced .-> TR
    TR --> DASH[Dashboard]
    O <--> ST[State - SQLite]
```

## Components

- **Orchestrator** — decomposes an incoming task, routes subtasks to agents,
  persists progress to state so long-running tasks survive a restart.
- **Agents** — each subclasses `BaseAgent`'s step-limited loop and only supplies
  its own system prompt and tool set.
- **Tool registry** — single source of truth for tool JSON schemas (sent to the
  LLM) and the Python callables that actually run when a tool is invoked.
- **Guardrails** — validates every action *before* execution: allow-listed
  filesystem paths, forbidden shell commands, per-task step ceilings.
- **Tracing** — logs every tool call with tokens, cost, and latency; feeds the
  dashboard and the eval harness's cost metrics.
- **Eval harness** — golden dataset of tasks plus LLM-as-judge scoring, so the
  project demonstrates production-readiness rather than just a demo.

## Why a provider-agnostic LLM client

`llm_client.py` normalizes Gemini's and Anthropic's tool-use APIs into one
`LLMResponse` shape. Agents only ever call `call_with_tools()` — they don't
know or care which provider answered. This means:

- Development happens for free against a Gemini key.
- Switching to Anthropic (e.g. for evaluation runs, or a cost/quality
  comparison) is a one-line env var change, not a rewrite.

## Resilience, guardrails & tracing (Phase 1-2)

Giving an LLM real tool access — even just to one machine, for one agent —
means a few failure modes need handling before the happy path matters:

- **Rate limits & transient errors.** `call_with_tools` retries on HTTP 429
  and 5xx with exponential backoff + jitter, capped at a small number of
  retries. A 429 from a daily quota being exhausted won't resolve itself by
  waiting a few seconds, so this fails fast after `MAX_RETRIES` rather than
  hammering the API.
- **Token/cost budget**, tracked separately from step count — a single large
  tool result (e.g. reading a big file) can blow a budget without using many
  steps.
- **Task-level wall-clock timeout**, independent of step count — bounds how
  long one task can run in total, on top of the per-step model call.
- **Repetition guard** — if the model requests the exact same tool call
  several times in a row, the loop stops instead of burning budget on a
  stuck agent.
- **Tool execution errors are isolated.** A failing tool (missing file, bad
  args, command timeout) becomes an error result sent back to the model,
  not a crashed task — the model gets a chance to adapt and, in practice,
  often does (see the fizzbuzz example in the project history: the agent
  ran pytest, read the failure, fixed the bug, and re-ran tests on its own).
- **Sandboxing**, hardcoded into the tools themselves. Filesystem tools
  resolve every path and reject anything outside the sandbox root
  (including `../` traversal and absolute paths). The exec tool uses
  `shell=False` with `shlex.split` (no shell-metacharacter injection), runs
  with `cwd` fixed to the sandbox, and always has a timeout.
- **Guardrails** (`guardrails.py`), a separate *configurable policy* layer on
  top of the hardcoded sandboxing above — checked before a tool runs, not
  baked into the tool implementation. A deny-list of regex patterns blocks
  commands like `rm`, `sudo`, `git push`, `pip install`; a size limit blocks
  oversized writes. Either one short-circuits before the real tool function
  is ever called.
- **Tool-scope enforcement.** Each agent only sees schemas for its own
  declared `tool_names`, but the shared tool registry holds every tool any
  agent might use. `base_agent.py` defensively re-checks that a requested
  tool name is actually in *this* agent's scope before executing it — a
  hallucinated or scope-confused call is rejected the same way a guardrail
  violation is, not assumed-safe just because it parsed.
- **Tracing** (`tracing.py`) — every LLM call and every tool call (including
  blocked and errored ones) is written as one JSON line to
  `traces/<task_id>.jsonl`: tokens, latency, blocked/error flags, truncated
  input/output. This is what turns "the agent did something for 5 steps and
  I have no idea what" into an inspectable record — and what Phase 5's eval
  harness will pull cost/latency numbers from. $ cost is computed only if
  you fill in current provider pricing via env vars; it's not hardcoded
  here since prices change.


## Status

This document grows alongside the implementation. See the phase table in the
[README](../README.md) for current status.
