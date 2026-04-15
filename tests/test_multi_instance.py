from axis.multi_instance import FederatedCoordinator
from axis.telemetry import TelemetryObserver
from axis.types import (
    ExecutionTrace,
    Strategy,
    Task,
    TaskOutcome,
)


def test_candidate_procedure_promotes_after_three_instances():
    coord = FederatedCoordinator(
        validation_threshold=0.7, min_instances_to_validate=3
    )
    proc = coord.propose_candidate(
        instance_id="i1",
        name="debug_python",
        task_class="code_debug",
        steps=[{"action": "run_tests"}],
    )
    assert proc.status == "candidate"

    for inst in ("i1", "i2", "i3"):
        coord.report_usage(
            instance_id=inst,
            procedure_id=proc.procedure_id,
            success_rate=0.85,
            sample_size=20,
        )
    promoted = coord.check_validations()
    assert len(promoted) == 1
    assert proc.status == "validated"


def test_candidate_not_promoted_without_enough_instances():
    coord = FederatedCoordinator(min_instances_to_validate=3)
    proc = coord.propose_candidate("i1", "p", "chat", [{}])
    coord.report_usage("i1", proc.procedure_id, 0.9, 50)
    coord.report_usage("i2", proc.procedure_id, 0.9, 50)
    assert coord.check_validations() == []
    assert proc.status == "candidate"


def test_low_success_rate_reports_do_not_promote():
    coord = FederatedCoordinator(
        validation_threshold=0.7, min_instances_to_validate=3
    )
    proc = coord.propose_candidate("i1", "p", "chat", [{}])
    for inst in ("i1", "i2", "i3"):
        coord.report_usage(inst, proc.procedure_id, 0.4, 20)
    assert coord.check_validations() == []


def test_local_override_logged_for_review():
    coord = FederatedCoordinator()
    proc = coord.propose_candidate("i1", "p", "chat", [{}])
    coord.record_local_override("i1", proc.procedure_id, "fails on our data")
    reviews = coord.overrides_for_review()
    assert len(reviews) == 1
    assert reviews[0].reason == "fails on our data"


def test_aggregate_self_model_merges_per_instance_telemetry():
    obs_a = TelemetryObserver()
    obs_b = TelemetryObserver()

    # Instance A: chat win rate 1.0 on 3 samples
    for i in range(3):
        trace = ExecutionTrace(
            task_id=f"a{i}",
            task_class="chat",
            strategy=Strategy.ANALYTICAL,
            execution_tokens=50,
            total_tokens=50,
            outcome=TaskOutcome.SUCCESS,
        )
        obs_a.record_trace(trace, Task(task_id=f"a{i}", input="x", task_class="chat"))

    # Instance B: chat win rate 0.5 on 4 samples
    for i in range(4):
        outcome = TaskOutcome.SUCCESS if i < 2 else TaskOutcome.FAILURE
        trace = ExecutionTrace(
            task_id=f"b{i}",
            task_class="chat",
            strategy=Strategy.ANALYTICAL,
            execution_tokens=50,
            total_tokens=50,
            outcome=outcome,
            error="x" if outcome == TaskOutcome.FAILURE else None,
        )
        obs_b.record_trace(trace, Task(task_id=f"b{i}", input="x", task_class="chat"))

    coord = FederatedCoordinator()
    agg = coord.aggregate_self_model({"A": obs_a, "B": obs_b})
    chat = agg["capabilities"]["chat"]
    # Weighted average: (1.0*3 + 0.5*4) / 7 ≈ 0.714
    assert abs(chat["win_rate"] - 0.7143) < 0.01
    assert chat["sample_size"] == 7
    assert chat["contributing_instances"] == 2
    assert agg["instances"] == 2
