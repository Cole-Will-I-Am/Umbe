import pytest

from axis.world_model import ActionSchema, TypedState, WorldModel


def _debug_schemas() -> list[ActionSchema]:
    return [
        ActionSchema(
            action="run_debugger",
            preconditions=["code_file_exists", "runtime_available"],
            postconditions_on_success=["diagnosis=null_pointer_line_42"],
            estimated_latency_ms=500,
        ),
        ActionSchema(
            action="apply_fix",
            preconditions=["diagnosis=null_pointer_line_42"],
            postconditions_on_success=["code=patched"],
            estimated_latency_ms=200,
        ),
        ActionSchema(
            action="run_tests",
            preconditions=["code=patched"],
            postconditions_on_success=["tests=passing"],
            postconditions_on_failure=["tests=failing"],
            estimated_latency_ms=5000,
        ),
    ]


def test_typed_state_predicates():
    s = TypedState(values={"code": "buggy", "tests": "failing"})
    assert s.satisfies("code=buggy")
    assert not s.satisfies("code=patched")
    assert s.satisfies("code")  # truthy
    assert s.satisfies("!missing_key")
    assert s.satisfies("code!=patched")


def test_typed_state_apply_returns_new_state():
    s = TypedState(values={"code": "buggy"})
    s2 = s.apply(["code=patched", "tests_run"])
    assert s2.values["code"] == "patched"
    assert s2.values["tests_run"] is True
    # Original untouched
    assert s.values["code"] == "buggy"
    # Negation deletes
    s3 = s.apply(["!code"])
    assert "code" not in s3.values


def test_plan_feasibility_passes_for_valid_sequence():
    wm = WorldModel(_debug_schemas())
    start = TypedState(values={"code_file_exists": True, "runtime_available": True})
    assert wm.plan_is_feasible(
        start, ["run_debugger", "apply_fix", "run_tests"]
    )


def test_plan_feasibility_fails_when_precondition_unmet():
    wm = WorldModel(_debug_schemas())
    # Missing runtime_available
    start = TypedState(values={"code_file_exists": True})
    assert not wm.plan_is_feasible(start, ["run_debugger"])


def test_simulate_sequence_returns_transitions_up_to_failure():
    wm = WorldModel(_debug_schemas())
    start = TypedState(values={"code_file_exists": True})
    transitions = wm.simulate_sequence(
        start, ["run_debugger", "apply_fix", "run_tests"]
    )
    assert len(transitions) == 1
    assert not transitions[0].succeeded


def test_counterfactual_bounded_by_max_candidates():
    wm = WorldModel(_debug_schemas())
    start = TypedState(values={"code_file_exists": True, "runtime_available": True})
    candidates = [
        ["run_debugger", "apply_fix", "run_tests"],
        ["run_debugger", "apply_fix"],
        ["run_tests"],  # infeasible
    ]
    results = wm.counterfactual_sequences(start, candidates, max_candidates=2)
    assert len(results) == 2
    assert results[0][1] is True
    # Estimated latency is the sum of schema latencies
    assert results[0][2] == 500 + 200 + 5000
