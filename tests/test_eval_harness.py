from axis.eval_harness import EvalCase, EvalHarness, EvalReport
from axis.types import (
    ExecutionTrace,
    Stakes,
    Strategy,
    Task,
    TaskOutcome,
)


def _mk_trace(outcome=TaskOutcome.SUCCESS, tokens=100, conf=0.8):
    return ExecutionTrace(
        task_id="t",
        task_class="general",
        strategy=Strategy.ANALYTICAL,
        execution_tokens=tokens,
        total_tokens=tokens,
        outcome=outcome,
        confidence_predicted=conf,
    )


def _fixed_runner(outcome_by_case: dict[str, TaskOutcome], tokens: int = 100):
    def run(task: Task) -> ExecutionTrace:
        return _mk_trace(
            outcome=outcome_by_case.get(task.task_id, TaskOutcome.SUCCESS),
            tokens=tokens,
        )

    return run


def _cases(n: int = 4) -> list[EvalCase]:
    cases = []
    for i in range(n):
        cases.append(
            EvalCase(
                case_id=f"target_{i}",
                task=Task(task_id=f"target_{i}", input="x"),
                eval_type="target_task",
                expected_outcome="success",
            )
        )
    for i in range(n):
        cases.append(
            EvalCase(
                case_id=f"adj_{i}",
                task=Task(task_id=f"adj_{i}", input="x"),
                eval_type="adjacent_task",
                expected_outcome="success",
            )
        )
    return cases


def test_harness_passes_when_target_improves_no_regression():
    harness = EvalHarness(cases=_cases())

    baseline = _fixed_runner(
        {f"target_{i}": TaskOutcome.FAILURE for i in range(3)},  # 1/4 pass
    )
    candidate = _fixed_runner({})  # 4/4 pass

    report = harness.run_suite(baseline, candidate)
    assert isinstance(report, EvalReport)
    assert report.target_improved
    assert report.regressions == []
    assert report.passed


def test_harness_fails_on_adjacent_regression():
    harness = EvalHarness(cases=_cases(n=10))

    baseline = _fixed_runner({})  # all pass
    # Candidate: target still all pass, but 8/10 adjacent fail
    candidate_failures = {
        f"target_{i}": TaskOutcome.FAILURE for i in range(1)  # barely improves
    }
    candidate_failures.update({f"adj_{i}": TaskOutcome.FAILURE for i in range(8)})
    candidate = _fixed_runner(candidate_failures)

    report = harness.run_suite(baseline, candidate)
    assert "adjacent_task" in report.regressions
    assert not report.passed


def test_harness_fails_on_cost_regression():
    harness = EvalHarness(cases=_cases(), cost_regression_pct=0.1)
    baseline = _fixed_runner({f"target_{i}": TaskOutcome.FAILURE for i in range(3)}, tokens=100)
    candidate = _fixed_runner({}, tokens=200)  # passes everything but 2x cost
    report = harness.run_suite(baseline, candidate)
    assert report.cost_regression
    assert not report.passed


def test_harness_calibration_case_tolerance():
    harness = EvalHarness(
        cases=[
            EvalCase(
                case_id="cal_1",
                task=Task(task_id="cal_1", input="x"),
                eval_type="calibration",
                metadata={"calibration_tolerance": 0.1},
            )
        ]
    )

    def well_calibrated(task):
        return _mk_trace(conf=1.0)  # predicted 1.0, outcome success → error 0

    def miscalibrated(task):
        return _mk_trace(conf=0.2)  # predicted 0.2, outcome success → error 0.8

    # Baseline well-calibrated, candidate miscalibrated — candidate fails
    report = harness.run_suite(well_calibrated, miscalibrated)
    # Since this is a calibration-only suite, target_task_report absent →
    # target_improved = False, so passed = False.
    assert not report.passed
