"""
Tool registry: maps tool name -> (schema, callable).

Agents declare which tool *names* they're allowed to use; the registry is
the single source of truth both for the JSON schemas sent to the LLM and
for the actual Python functions that run when the model calls them.

Implemented in Phase 1.
"""
