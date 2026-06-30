"""
Live trace dashboard.

Reads the exact same JSONL trace files (tracing.py) and SQLite state
(state.py) every other part of this project already writes — no separate
data pipeline, no new storage format. If you've run any agent or the
orchestrator at all, there's something here to look at.

Run with:
    streamlit run dashboard/app.py

Auto-refreshes its data every 5 seconds (st.cache_data ttl), so you can
leave it open in a second window while running tasks in a terminal and
watch them show up.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

# Makes `streamlit run dashboard/app.py` work directly, with no PYTHONPATH setup.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402  (streamlit depends on pandas; no extra install needed)
import streamlit as st  # noqa: E402

from orchestrator.config import get_config  # noqa: E402

st.set_page_config(page_title="AgentOps Dashboard", layout="wide")

cfg = get_config()
TRACE_DIR = Path(cfg.trace_dir)
DB_PATH = Path(cfg.db_path)


@st.cache_data(ttl=5)
def load_trace_records() -> list[dict]:
    records = []
    if TRACE_DIR.is_dir():
        for path in sorted(TRACE_DIR.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


@st.cache_data(ttl=5)
def load_orchestrator_state() -> tuple[list[tuple], list[tuple]]:
    if not DB_PATH.is_file():
        return [], []
    conn = sqlite3.connect(DB_PATH)
    try:
        tasks = conn.execute(
            "SELECT task_id, original_task, status, created_at FROM tasks ORDER BY created_at DESC"
        ).fetchall()
        subtasks = conn.execute(
            "SELECT task_id, subtask_index, agent, description, status, result_json "
            "FROM subtasks ORDER BY task_id, subtask_index"
        ).fetchall()
    finally:
        conn.close()
    return tasks, subtasks


def _event_icon(record: dict) -> str:
    if record["type"] == "llm_error":
        return "💥"
    if record["type"] == "task_end":
        return "🏁" if record.get("stopped_reason") == "done" else "⚠️"
    if record["type"] == "tool_call":
        if record.get("blocked"):
            return "🚫"
        if record.get("is_error"):
            return "❌"
        return "🔧"
    return "🧠"  # llm_call


def _format_event(record: dict) -> str:
    icon = _event_icon(record)
    rtype = record["type"]

    if rtype == "llm_call":
        return (
            f"{icon} step {record['step']} — LLM call · {record['latency_seconds']}s · "
            f"in={record['input_tokens']} out={record['output_tokens']}"
        )
    if rtype == "llm_error":
        return f"{icon} step {record['step']} — LLM error: {record['error']}"
    if rtype == "tool_call":
        status = (
            "blocked" if record.get("blocked") else ("error" if record.get("is_error") else "ok")
        )
        return (
            f"{icon} step {record['step']} — {record['name']}({record.get('arguments', '')}) "
            f"[{status}] → {str(record.get('output', ''))[:300]}"
        )
    if rtype == "task_end":
        return (
            f"{icon} finished — {record['stopped_reason']} after {record['steps_taken']} steps "
            f"(in={record.get('total_input_tokens', 0)} out={record.get('total_output_tokens', 0)})"
        )
    return f"{icon} {rtype}"


st.title("AgentOps Orchestrator — Live Dashboard")
st.caption(f"Reading traces from `{TRACE_DIR}` and state from `{DB_PATH}`")

records = load_trace_records()
task_ends = [r for r in records if r.get("type") == "task_end"]

if not task_ends:
    st.info(
        "No traces found yet. Run something to see data here — e.g.\n\n"
        "`python scripts/run_swe_agent.py \"create hello.txt containing 'hi'\"`"
    )
    st.stop()

# --- Overview metrics ------------------------------------------------------

total_tasks = len(task_ends)
success_count = sum(1 for r in task_ends if r.get("stopped_reason") == "done")
total_in = sum(r.get("total_input_tokens", 0) for r in task_ends)
total_out = sum(r.get("total_output_tokens", 0) for r in task_ends)
total_cost = sum(r.get("estimated_cost_usd", 0.0) for r in task_ends)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Tasks run", total_tasks)
col2.metric("Success rate", f"{success_count / total_tasks:.0%}")
col3.metric("Input tokens", f"{total_in:,}")
col4.metric("Output tokens", f"{total_out:,}")
col5.metric("Est. cost", f"${total_cost:.4f}" if total_cost else "$0 (pricing not set)")

st.divider()

# --- Task table -------------------------------------------------------------

st.subheader("Recent tasks")

df = pd.DataFrame(task_ends)
df = df.sort_values("timestamp", ascending=False)
display_cols = [
    c
    for c in [
        "task_id",
        "stopped_reason",
        "steps_taken",
        "total_input_tokens",
        "total_output_tokens",
    ]
    if c in df.columns
]
st.dataframe(df[display_cols], width="stretch", hide_index=True)

# --- Task detail ------------------------------------------------------------

st.subheader("Task detail")
task_ids = df["task_id"].tolist()
selected = st.selectbox("Select a task_id", task_ids)

if selected:
    timeline = sorted(
        (r for r in records if r.get("task_id") == selected),
        key=lambda r: r.get("timestamp", 0),
    )
    for record in timeline:
        st.text(_format_event(record))

st.divider()

# --- Orchestrator plans ------------------------------------------------------

st.subheader("Orchestrator plans")

tasks, subtasks = load_orchestrator_state()
if not tasks:
    st.caption("No orchestrator runs yet (single-agent runs don't use SQLite state).")
else:
    for task_id, original_task, status, created_at in tasks:
        with st.expander(f"{task_id} — {original_task[:80]} [{status}]"):
            for st_task_id, idx, agent, description, st_status, _ in subtasks:
                if st_task_id == task_id:
                    st.write(f"{idx}. **[{agent}]** {description} — *{st_status}*")
