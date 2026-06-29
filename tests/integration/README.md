# Integration tests

This directory ended up unused — what it was meant to cover (real calls
across module boundaries, e.g. an agent actually executing a tool and
feeding the result back to the LLM client) turned out to fit more
naturally as specific tests within `tests/unit/`, using the real tool
implementations (not stand-ins) through a real agent loop with only the
LLM call itself mocked. See e.g.
`test_base_agent.py::test_real_write_file_respects_the_calling_agents_config_not_the_global_one`
and the equivalent in `test_eval_runner.py` — both exist specifically
because a bug lived in the seam between "agent has a Config" and "the
real tool functions it calls respect it," and a pure-unit test on either
side alone wouldn't have caught it. See
[docs/architecture.md](../../docs/architecture.md) for that story.
