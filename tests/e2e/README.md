# End-to-end tests

This directory ended up unused — what it was meant to cover (full
task-in, result-out runs against real APIs) is what `eval/` actually is:
`eval/golden_dataset.jsonl` + `eval/run_eval.py`, with outcome
verification instead of pass/fail assertions, and a markdown report
instead of pytest output. It's a different enough shape (makes real API
calls, isn't run by default in CI, costs real tokens) that it stayed a
separate system rather than living under `tests/`. See
[eval/README.md](../../eval/README.md).
