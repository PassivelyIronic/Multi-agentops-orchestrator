"""
Generic agent loop: step limit, timeout, retry.

Concrete agents (SweAgent, TesterAgent, OnCallAgent, PmAgent) subclass this
and only provide their own system prompt and tool set — the loop itself
(call model -> execute requested tool -> feed result back -> repeat until
done or step limit hit) lives here exactly once.

Implemented in Phase 1.
"""
