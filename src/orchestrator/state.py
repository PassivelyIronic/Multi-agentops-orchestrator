"""
Task state persistence (SQLite).

Stores task status, current step, and step history so long-running tasks can
be checkpointed and resumed after a restart instead of living only in process
memory. This is what makes "16+ hour agent run" a property of the
architecture rather than something you have to babysit live.

Implemented in Phase 3.
"""
