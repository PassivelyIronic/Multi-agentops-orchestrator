"""
Unit tests for TraceLogger — writing/reading real JSONL files against
tmp_path, since the whole point is verifying what actually lands on disk.
"""

from __future__ import annotations

import json

from orchestrator.config import Config
from orchestrator.tracing import TraceLogger, new_task_id


def _fake_config(trace_dir, **overrides) -> Config:
    defaults = dict(
        llm_provider="gemini",
        gemini_api_key="fake",
        anthropic_api_key="fake",
        trace_dir=str(trace_dir),
    )
    defaults.update(overrides)
    return Config(**defaults)


def _read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_new_task_id_is_unique():
    assert new_task_id() != new_task_id()


def test_log_llm_call_writes_expected_fields(tmp_path):
    tracer = TraceLogger(task_id="t1", config=_fake_config(tmp_path))
    tracer.log_llm_call(step=1, latency_seconds=0.25, input_tokens=100, output_tokens=20)

    records = _read_records(tmp_path / "t1.jsonl")
    assert len(records) == 1
    assert records[0]["type"] == "llm_call"
    assert records[0]["input_tokens"] == 100
    assert records[0]["output_tokens"] == 20
    assert records[0]["latency_seconds"] == 0.25
    assert records[0]["task_id"] == "t1"


def test_log_tool_call_truncates_large_output(tmp_path):
    tracer = TraceLogger(task_id="t2", config=_fake_config(tmp_path))
    huge_output = "x" * 5_000

    tracer.log_tool_call(
        step=1,
        name="read_file",
        arguments={"path": "big.txt"},
        output=huge_output,
        is_error=False,
        blocked=False,
        latency_seconds=0.1,
    )

    record = _read_records(tmp_path / "t2.jsonl")[0]
    assert len(record["output"]) < len(huge_output)
    assert record["output"].endswith("...[truncated]")


def test_log_task_end_records_summary(tmp_path):
    tracer = TraceLogger(task_id="t3", config=_fake_config(tmp_path))
    tracer.log_task_end("done", steps_taken=3, total_input_tokens=500, total_output_tokens=80)

    record = _read_records(tmp_path / "t3.jsonl")[0]
    assert record["type"] == "task_end"
    assert record["stopped_reason"] == "done"
    assert record["steps_taken"] == 3


def test_cost_defaults_to_zero_without_configured_pricing(tmp_path):
    tracer = TraceLogger(task_id="t4", config=_fake_config(tmp_path))
    tracer.log_llm_call(
        step=1, latency_seconds=0.1, input_tokens=1_000_000, output_tokens=1_000_000
    )

    record = _read_records(tmp_path / "t4.jsonl")[0]
    assert record["estimated_cost_usd"] == 0.0


def test_cost_is_computed_when_pricing_is_configured(tmp_path):
    cfg = _fake_config(
        tmp_path, gemini_input_price_per_million=1.0, gemini_output_price_per_million=2.0
    )
    tracer = TraceLogger(task_id="t5", config=cfg)
    tracer.log_llm_call(
        step=1, latency_seconds=0.1, input_tokens=1_000_000, output_tokens=1_000_000
    )

    record = _read_records(tmp_path / "t5.jsonl")[0]
    assert record["estimated_cost_usd"] == 3.0  # 1*1.0 + 1*2.0


def test_multiple_calls_append_rather_than_overwrite(tmp_path):
    tracer = TraceLogger(task_id="t6", config=_fake_config(tmp_path))
    tracer.log_llm_call(step=1, latency_seconds=0.1, input_tokens=10, output_tokens=5)
    tracer.log_llm_call(step=2, latency_seconds=0.1, input_tokens=10, output_tokens=5)

    records = _read_records(tmp_path / "t6.jsonl")
    assert len(records) == 2
