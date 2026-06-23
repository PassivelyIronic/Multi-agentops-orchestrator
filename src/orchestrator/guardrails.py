"""
Guardrails: validates agent actions before execution.

Examples: filesystem writes restricted to an allow-listed directory, forbidden
shell commands (e.g. `git push --force`, `rm -rf`), per-task step ceilings.
Guardrails run *before* a tool executes, separate from the step-limit logic
in base_agent.py, so the two can be tested independently.

Implemented in Phase 2.
"""
