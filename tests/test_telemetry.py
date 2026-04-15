import json
from pathlib import Path

import pytest

from axis.telemetry import (
    DEFAULT_WINDOWS,
    SCHEMA_VERSION,
    StructuredSink,
    TelemetryObserver,
    TelemetryRecord,
    _percentile,
)
from axis.types import (
    Budget,
    ExecutionTrace,
    Gates,
    Plan,
    PlanStep,
    Stakes,
    StepResult,
    Strategy,
    Task,
    TaskOutcome,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_trace(
    task_id: str = "t",
    task_class: str = "general",
    strategy: Strategy = Strategy.ANALYTICAL,
    outcome: TaskOutcome = TaskOutcome.SUCCESS,
    execution_tokens: int = 100,
    latency_ms: int = 500,
    tool_calls: list[dict] | None = None,
    replans: int = 0,
    error: str | None = None,
    predicted_confidence: float | None = None,
) -> ExecutionTrace:
    step = StepResult(
        step_id="s1",
        output="ok",
        tokens_used=execution_tokens,
        tool_calls=tool_calls or [],
        success=outcome != TaskOutcome.FAILURE,
        error=error if outcome == TaskOutcome.FAILURE else None,
    )
    trace = ExecutionTrace(
        task_id=task_id,
        task_class=task_class,
        strategy=strategy,
        step_results=[step],
        execution_tokens=execution_tokens,
        total_tokens=execution_tokens,
        replanning_events=replans,
        outcome=outcome,
        error=error,
        confidence_predicted=predicted_confidence,
    )
    # Fake a fixed latency so percentile tests are deterministic
    trace.start_time = 0.0
    trace.end_time = latency_ms / 1000.0
    return trace


def _make_task(task_class: str = "general") -> Task:
    return Task(input="x", task_class=task_class)


# ---------------------------------------------------------------------------
# TelemetryRecord
# ---------------------------------------------------------------------------
def test_record_from_trace_copies_fields():
    trace = _make_trace(task_class="analysis", execution_tokens=321)
    rec = TelemetryRecord.from_trace(trace, _make_task("analysis"))
    assert rec.task_class == "analysis"
    assert rec.strategy == Strategy.ANALYTICAL.value
    assert rec.execution_tokens == 321
    assert rec.resource_cost_tokens == 321
    assert rec.outcome == TaskOutcome.SUCCESS.value
    assert rec.schema_version == SCHEMA_VERSION


def test_record_serializes_to_json():
    trace = _make_trace()
    rec = TelemetryRecord.from_trace(trace, _make_task())
    payload = json.loads(rec.to_json())
    assert payload["task_id"] == "t"
    assert payload["strategy"] == "analytical"


# ---------------------------------------------------------------------------
# Observer — ingest + metrics
# ---------------------------------------------------------------------------
def test_ingest_updates_win_rate_per_class():
    obs = TelemetryObserver()
    for outcome in [TaskOutcome.SUCCESS, TaskOutcome.SUCCESS, TaskOutcome.FAILURE]:
        obs.record_trace(
            _make_trace(task_class="code_debug", outcome=outcome),
            _make_task("code_debug"),
        )
    m = obs.metrics()
    assert pytest.approx(m["win_rate_by_class"]["code_debug"], rel=1e-6) == 2 / 3


def test_strategy_bias_distribution():
    obs = TelemetryObserver()
    for _ in range(3):
        obs.record_trace(
            _make_trace(strategy=Strategy.ANALYTICAL), _make_task()
        )
    for _ in range(1):
        obs.record_trace(
            _make_trace(strategy=Strategy.TOOL_FIRST), _make_task()
        )
    bias = obs.metrics()["strategy_bias"]
    assert pytest.approx(bias["analytical"], rel=1e-6) == 0.75
    assert pytest.approx(bias["tool_first"], rel=1e-6) == 0.25


def test_tool_success_rate_per_tool():
    obs = TelemetryObserver()
    obs.record_trace(
        _make_trace(
            tool_calls=[
                {"tool": "retrieval", "success": True},
                {"tool": "retrieval", "success": False},
                {"tool": "calculator", "success": True},
            ]
        ),
        _make_task(),
    )
    rates = obs.metrics()["tool_success_rate"]
    assert pytest.approx(rates["retrieval"], rel=1e-6) == 0.5
    assert pytest.approx(rates["calculator"], rel=1e-6) == 1.0


def test_avg_cost_by_class():
    obs = TelemetryObserver()
    for cost in (100, 200, 300):
        obs.record_trace(
            _make_trace(task_class="analysis", execution_tokens=cost),
            _make_task("analysis"),
        )
    avg = obs.metrics()["avg_cost_by_class"]["analysis"]
    assert avg == 200.0


def test_replanning_rate():
    obs = TelemetryObserver()
    obs.record_trace(_make_trace(replans=0), _make_task())
    obs.record_trace(_make_trace(replans=0), _make_task())
    obs.record_trace(_make_trace(replans=1), _make_task())
    obs.record_trace(_make_trace(replans=2), _make_task())
    rate = obs.metrics()["replanning_rate"]
    assert rate == 0.5


def test_latency_distribution_percentiles():
    obs = TelemetryObserver()
    for ms in (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000):
        obs.record_trace(
            _make_trace(task_class="general", latency_ms=ms),
            _make_task("general"),
        )
    dist = obs.metrics()["latency_distribution_by_class"]["general"]
    assert dist["count"] == 10
    assert dist["p50"] == pytest.approx(550.0)
    assert dist["p90"] == pytest.approx(910.0)
    assert dist["p50"] < dist["p90"] < dist["p95"] < dist["p99"]


def test_failure_mode_frequency_clusters_errors():
    obs = TelemetryObserver()
    for err in ("oom", "oom", "timeout"):
        obs.record_trace(
            _make_trace(outcome=TaskOutcome.FAILURE, error=err), _make_task()
        )
    freq = obs.metrics()["failure_mode_frequency"]
    assert pytest.approx(freq["oom"], rel=1e-6) == 2 / 3
    assert pytest.approx(freq["timeout"], rel=1e-6) == 1 / 3


def test_calibration_error_only_when_predicted_available():
    obs = TelemetryObserver()
    # No predicted confidence → None
    obs.record_trace(_make_trace(), _make_task())
    assert obs.metrics()["confidence_calibration_error"] is None
    # Add predicted confidences
    obs.record_trace(
        _make_trace(
            outcome=TaskOutcome.SUCCESS, predicted_confidence=0.8
        ),
        _make_task(),
    )
    obs.record_trace(
        _make_trace(
            outcome=TaskOutcome.FAILURE, predicted_confidence=0.9
        ),
        _make_task(),
    )
    ce = obs.metrics()["confidence_calibration_error"]
    # |0.8 - 1| + |0.9 - 0| = 1.1; mean = 0.55
    assert pytest.approx(ce, rel=1e-6) == 0.55


def test_missing_optional_fields_are_tolerated():
    obs = TelemetryObserver()
    obs.record_trace(_make_trace(), _make_task())
    m = obs.metrics()
    # These degrade to None / empty, not fabricated values
    assert m["confidence_calibration_error"] is None
    assert m["strategy_bias"]["analytical"] == 1.0


# ---------------------------------------------------------------------------
# Ring buffer bounds
# ---------------------------------------------------------------------------
def test_win_rate_ring_buffer_bounded():
    obs = TelemetryObserver()
    cap = DEFAULT_WINDOWS["win_rate_per_class"]
    # Fill with failures then flood with successes
    for _ in range(cap):
        obs.record_trace(
            _make_trace(task_class="chat", outcome=TaskOutcome.FAILURE),
            _make_task("chat"),
        )
    for _ in range(cap):
        obs.record_trace(
            _make_trace(task_class="chat", outcome=TaskOutcome.SUCCESS),
            _make_task("chat"),
        )
    # Only the last `cap` entries count — all successes
    assert obs.metrics()["win_rate_by_class"]["chat"] == 1.0


# ---------------------------------------------------------------------------
# Structured sink
# ---------------------------------------------------------------------------
def test_structured_sink_appends_jsonl(tmp_path: Path):
    sink_path = tmp_path / "telemetry.jsonl"
    sink = StructuredSink(sink_path)
    obs = TelemetryObserver(sink=sink)
    obs.record_trace(_make_trace(task_id="a"), _make_task())
    obs.record_trace(_make_trace(task_id="b"), _make_task())
    lines = sink_path.read_text().strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert {p["task_id"] for p in parsed} == {"a", "b"}


def test_structured_sink_read_all_roundtrip(tmp_path: Path):
    sink = StructuredSink(tmp_path / "t.jsonl")
    obs = TelemetryObserver(sink=sink)
    obs.record_trace(_make_trace(task_id="x"), _make_task())
    data = sink.read_all()
    assert len(data) == 1
    assert data[0]["task_id"] == "x"


# ---------------------------------------------------------------------------
# Self-Model snapshot (§6: only observer writes self-model)
# ---------------------------------------------------------------------------
def test_self_model_snapshot_builds_from_telemetry():
    obs = TelemetryObserver()
    for _ in range(4):
        obs.record_trace(
            _make_trace(
                task_class="code_debug",
                strategy=Strategy.RETRIEVAL_FIRST,
                execution_tokens=500,
                outcome=TaskOutcome.SUCCESS,
            ),
            _make_task("code_debug"),
        )
    obs.record_trace(
        _make_trace(
            task_class="code_debug",
            strategy=Strategy.RETRIEVAL_FIRST,
            outcome=TaskOutcome.FAILURE,
            error="bad_fix",
        ),
        _make_task("code_debug"),
    )
    snap = obs.self_model_snapshot()

    cap = snap["capabilities"]["code_debug"]
    assert cap["win_rate"] == 0.8
    assert cap["sample_size"] == 5

    eff = snap["strategy_effectiveness"]["retrieval_first"]
    assert eff["win_rate_when_selected"] == 0.8
    assert eff["sample_size"] == 5

    assert snap["resource_profile"]["avg_tokens_per_task"] > 0
    assert any(fm["pattern"] == "bad_fix" for fm in snap["failure_modes"])
    assert snap["update_source"].startswith("telemetry_observer:")


def test_should_refresh_self_model_every_n_tasks():
    obs = TelemetryObserver(self_model_refresh_every=5)
    for i in range(4):
        obs.record_trace(_make_trace(task_id=str(i)), _make_task())
        assert obs.should_refresh_self_model() is False
    obs.record_trace(_make_trace(task_id="5"), _make_task())
    assert obs.should_refresh_self_model() is True


def test_drift_flag_triggers_refresh():
    obs = TelemetryObserver(self_model_refresh_every=10_000)
    # Establish a steady baseline of successes, then flood with failures
    for _ in range(30):
        obs.record_trace(
            _make_trace(task_class="steady", outcome=TaskOutcome.SUCCESS),
            _make_task("steady"),
        )
    # The last 10 are still successes → no drift yet
    assert obs.drift_flags() == set()
    for _ in range(10):
        obs.record_trace(
            _make_trace(task_class="steady", outcome=TaskOutcome.FAILURE),
            _make_task("steady"),
        )
    flags = obs.drift_flags()
    assert "win_rate:steady" in flags
    assert obs.should_refresh_self_model() is True


# ---------------------------------------------------------------------------
# _percentile helper
# ---------------------------------------------------------------------------
def test_percentile_linear_interpolation():
    assert _percentile([1.0], 0.5) == 1.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert _percentile([10, 20, 30, 40, 50], 0.9) == pytest.approx(46.0)


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        _percentile([], 0.5)
