"""
Orchestrator: decomposes an incoming task into subtasks and routes each
subtask to the right specialized agent, checkpointing progress after every
subtask so a crashed or restarted process resumes instead of starting over.

Decomposition is one plain LLM call (no tools) asking for a JSON list of
{agent, description} subtasks — reuses call_with_tools with an empty tools
list rather than introducing a second LLM-calling path, so retries/backoff
apply here too. Routing is keyed by agent name string via AVAILABLE_AGENTS;
only "swe" is wired up for now — Phase 4 adds Tester/On-call/PM by adding
entries to that dict, with no other change needed here.

Resumability: if a task_id already has a saved plan, run() loads it instead
of re-decomposing (zero extra LLM cost) and continues from the first
subtask that isn't 'done' — including re-running a subtask that was left
'running' by a process that died mid-step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import state
from .agents.base_agent import AgentResult, BaseAgent
from .agents.swe_agent import SweAgent
from .config import Config, get_config
from .llm_client import call_with_tools
from .tracing import new_task_id

AVAILABLE_AGENTS: dict[str, type[BaseAgent]] = {
    "swe": SweAgent,
}

DECOMPOSE_SYSTEM_PROMPT = """\
You are a technical project orchestrator. Break the given task into an
ordered list of subtasks. Each subtask is handled independently by one
agent, so its description must be self-contained — the agent doing
subtask 2 cannot see what happened in subtask 1 except through files left
on disk.

Available agents: {agents}

Respond with ONLY a JSON array, no prose, no markdown code fences. Each
element: {{"agent": "<agent name>", "description": "<specific instructions>"}}.
If the task doesn't need splitting, return a single-element array.
"""


@dataclass
class Subtask:
    agent: str
    description: str


@dataclass
class OrchestratorResult:
    task_id: str
    subtask_results: list[tuple[Subtask, AgentResult]] = field(default_factory=list)
    # "done" | "subtask_failed" | "unknown_agent"
    stopped_reason: str = "done"


def run(task: str, config: Config | None = None, task_id: str | None = None) -> OrchestratorResult:
    cfg = config or get_config()
    task_id = task_id or new_task_id()

    existing = state.load_subtasks(task_id, cfg)
    if existing:
        subtasks = [Subtask(agent=r.agent, description=r.description) for r in existing]
        start_index = next((r.subtask_index for r in existing if r.status != "done"), len(existing))
    else:
        subtasks = _decompose(task, cfg)
        state.save_plan(task_id, task, subtasks, cfg)
        start_index = 0

    results: list[tuple[Subtask, AgentResult]] = []
    for index, subtask in enumerate(subtasks):
        if index < start_index:
            continue  # already done in a previous run — resuming past it

        agent_cls = AVAILABLE_AGENTS.get(subtask.agent)
        if agent_cls is None:
            state.mark_subtask_failed(task_id, index, f"Unknown agent: {subtask.agent}", cfg)
            return OrchestratorResult(task_id, results, "unknown_agent")

        state.mark_subtask_running(task_id, index, cfg)
        agent_result = agent_cls(config=cfg).run(subtask.description, task_id=f"{task_id}-{index}")
        results.append((subtask, agent_result))

        if agent_result.stopped_reason != "done":
            state.mark_subtask_failed(task_id, index, agent_result.stopped_reason, cfg)
            return OrchestratorResult(task_id, results, "subtask_failed")

        state.mark_subtask_done(task_id, index, agent_result.final_text or "", cfg)

    state.mark_task_status(task_id, "done", cfg)
    return OrchestratorResult(task_id, results, "done")


def _decompose(task: str, cfg: Config) -> list[Subtask]:
    agents = list(AVAILABLE_AGENTS.keys())
    system = DECOMPOSE_SYSTEM_PROMPT.format(agents=", ".join(agents))
    response = call_with_tools(
        messages=[{"role": "user", "content": task}], tools=[], system=system, config=cfg
    )
    return _parse_subtasks(response.text, agents, fallback_task=task)


def _parse_subtasks(
    raw_text: str | None, available_agents: list[str], fallback_task: str
) -> list[Subtask]:
    """
    Defensive JSON parsing: any failure mode (empty response, malformed
    JSON, wrong shape, unknown agent name) degrades to a single fallback
    subtask covering the whole original task — decomposition is a nice-to-
    have, not something that should ever crash the task outright.
    """
    fallback = [Subtask(agent=available_agents[0], description=fallback_task)]
    if not raw_text:
        return fallback

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return fallback

    if not isinstance(data, list) or not data:
        return fallback

    subtasks = []
    for item in data:
        if not isinstance(item, dict):
            continue
        agent = item.get("agent")
        description = item.get("description")
        if agent in available_agents and isinstance(description, str) and description.strip():
            subtasks.append(Subtask(agent=agent, description=description.strip()))

    return subtasks or fallback
