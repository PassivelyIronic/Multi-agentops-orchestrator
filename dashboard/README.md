# Dashboard

A Streamlit app for live trace viewing — reads the exact same JSONL trace
files (`tracing.py`) and SQLite state (`state.py`) every other part of
this project already writes. No separate data pipeline, no new storage
format: if you've run any agent or the orchestrator at all, there's
something here to look at.

## Run it

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` by default. Auto-refreshes its data every
5 seconds, so it's useful to leave open in a second window while running
tasks in a terminal.

## What it shows

- **Overview metrics** — tasks run, success rate, total input/output
  tokens, estimated cost (0 unless you've filled in pricing env vars).
- **Recent tasks table** — one row per `task_end` trace record.
- **Task detail timeline** — every event for a selected task_id in order:
  LLM calls (latency, tokens), tool calls (with a blocked/error/ok status
  icon), LLM errors, and the final outcome.
- **Orchestrator plans** — for any task that went through `orchestrator.py`,
  the subtask breakdown and each subtask's status, read from
  `orchestrator_state.db`.

## Notes for anyone editing this file

`@st.cache_data` on the loader functions takes no arguments — fine for
real usage (one long-lived server process, a fixed `.env`), but it means
its cache key doesn't account for `TRACE_DIR`/`DB_PATH` changing. If you
add tests that run `AppTest.from_file` multiple times in the same process
with different env vars (see `tests/unit/test_dashboard.py`), call
`st.cache_data.clear()` between runs or the previous test's data will leak
into the next one.
