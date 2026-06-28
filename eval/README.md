# Evaluation

A golden dataset of 10 tasks with mechanically-checkable or LLM-judged
success criteria, plus a runner that executes each one in an isolated
sandbox and produces a pass/fail report.

## Run it

```bash
python eval/run_eval.py                              # all 10 tasks
python eval/run_eval.py --ids swe_fizzbuzz,swe_bugfix # a subset
```

Writes `eval/report.md` (pass rate, tokens, per-task detail, failure
reasoning). Each task's sandbox/traces/state live under
`eval/runs/<task_id>/` — both are gitignored, regenerated on every run.

## Why this exists

`stopped_reason="done"` means the model stopped calling tools — it does
not mean the task was actually accomplished. See
[docs/architecture.md](../docs/architecture.md#eval-harness-phase-5) for
the real failure this caught during manual testing: an agent's `write_file`
call silently failed, the model gave up and reported success anyway, and
nothing downstream noticed until the next agent in the pipeline happened
to check. `swe_bugfix` in the golden dataset is built from that exact
scenario.

## Verification types

| Type | Checks |
|---|---|
| `file_exists` | a file is present in the task's sandbox |
| `file_contains` | a file exists and contains a substring |
| `file_unchanged` | a file's content is byte-identical to an expected string — used to verify guardrails actually hold, e.g. the tester agent didn't touch the implementation it was testing |
| `no_files_matching` | no file in the sandbox matches a glob pattern — e.g. the PM agent didn't write any `.py` files |
| `command_succeeds` | a shell command run inside the sandbox exits 0 (e.g. `pytest -q`) |
| `llm_judge` | qualitative criteria scored by an LLM call against a rubric — fallback only, not the default |

A task's `"verification"` field can be a single spec or a list (all must
pass). See `golden_dataset.jsonl` for real examples of each.

## Cost

A full run is ~25-35 LLM calls total. On OpenRouter's free tier (50
requests/day for an unfunded account), that's a meaningful chunk of a
day's budget — use `--ids` for a subset, or switch providers, if you're
tight on quota.
