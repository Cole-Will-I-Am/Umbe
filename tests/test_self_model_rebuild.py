"""Priority 6 — full Self-Model rebuild from telemetry."""

from axis.telemetry import TelemetryObserver
from axis.types import (
    ExecutionTrace,
    Strategy,
    Task,
    TaskOutcome,
)


def _trace(
    task_class: str,
    strategy: Strategy,
    outcome: TaskOutcome,
    predicted_confidence=None,
    error: str | None = None,
) -> ExecutionTrace:
    return ExecutionTrace(
        task_id="t",
        task_class=task_class,
        strategy=strategy,
        execution_tokens=100,
        total_tokens=100,
        outcome=outcome,
        error=error,
        confidence_predicted=predicted_confidence,
    )


def test_best_strategy_learned_per_class():
    obs = TelemetryObserver()
    # For legal_reasoning, retrieval_first wins 4/4, analytical wins 1/4
    for _ in range(4):
        obs.record_trace(
            _trace("legal_reasoning", Strategy.RETRIEVAL_FIRST, TaskOutcome.SUCCESS),
            Task(input="x", task_class="legal_reasoning"),
        )
    for i in range(4):
        outcome = TaskOutcome.SUCCESS if i == 0 else TaskOutcome.FAILURE
        obs.record_trace(
            _trace("legal_reasoning", Strategy.ANALYTICAL, outcome, error="bad"),
            Task(input="x", task_class="legal_reasoning"),
        )
    snap = obs.self_model_snapshot()
    assert snap["capabilities"]["legal_reasoning"]["best_strategy"] == "retrieval_first"


def test_per_class_calibration_error_computed():
    obs = TelemetryObserver()
    # Perfectly calibrated
    obs.record_trace(
        _trace("chat", Strategy.ANALYTICAL, TaskOutcome.SUCCESS, predicted_confidence=1.0),
        Task(input="x", task_class="chat"),
    )
    obs.record_trace(
        _trace("chat", Strategy.ANALYTICAL, TaskOutcome.FAILURE, predicted_confidence=0.0),
        Task(input="x", task_class="chat"),
    )
    snap = obs.self_model_snapshot()
    assert snap["capabilities"]["chat"]["confidence_calibration_error"] == 0.0


def test_known_failure_modes_populated_per_class():
    obs = TelemetryObserver()
    for err in ("timeout", "timeout", "oom"):
        obs.record_trace(
            _trace("code_debug", Strategy.ANALYTICAL, TaskOutcome.FAILURE, error=err),
            Task(input="x", task_class="code_debug"),
        )
    snap = obs.self_model_snapshot()
    modes = snap["capabilities"]["code_debug"]["known_failure_modes"]
    assert "timeout" in modes
    assert "oom" in modes
    # Most frequent should be first
    assert modes[0] == "timeout"


def test_insufficient_strategy_samples_do_not_set_best_strategy():
    obs = TelemetryObserver()
    obs.record_trace(
        _trace("chat", Strategy.ANALYTICAL, TaskOutcome.SUCCESS),
        Task(input="x", task_class="chat"),
    )
    snap = obs.self_model_snapshot()
    # Only 1 sample; min cluster size is 3 → no best_strategy selection
    assert snap["capabilities"]["chat"]["best_strategy"] is None
