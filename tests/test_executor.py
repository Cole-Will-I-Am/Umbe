from axis.backends.stub import StubBackend
from axis.executor import Executor, WorkingMemory
from axis.types import (
    Budget,
    Gates,
    Plan,
    PlanStep,
    Strategy,
    TaskOutcome,
)


def _plan(steps):
    return Plan(
        task_id="t1",
        steps=steps,
        strategy=Strategy.ANALYTICAL,
        budget=Budget(10_000, 1500, 5500, 1500, 1500),
        gates=Gates(),
    )


def _step(sid="s1", prompt="hello", max_tokens=128, depends_on=None, tool=None):
    return PlanStep(
        step_id=sid,
        action="reason",
        strategy=Strategy.ANALYTICAL,
        prompt=prompt,
        tool=tool,
        depends_on=depends_on or [],
        max_tokens=max_tokens,
    )


def test_single_step_success():
    ex = Executor(backend=StubBackend())
    trace = ex.execute_plan(_plan([_step()]))
    assert trace.outcome == TaskOutcome.SUCCESS
    assert trace.total_tokens > 0
    assert trace.step_results[0].success is True


def test_step_failure_aborts_trace():
    ex = Executor(backend=StubBackend(fail_on={"BOOM"}))
    trace = ex.execute_plan(_plan([_step(prompt="please BOOM here")]))
    assert trace.outcome == TaskOutcome.FAILURE
    assert trace.error is not None
    assert trace.step_results[0].success is False


def test_hard_token_ceiling_enforced():
    """Spec §1.1: hard token ceiling per execution call."""
    ex = Executor(backend=StubBackend())
    big_prompt = "x" * 10_000
    trace = ex.execute_plan(_plan([_step(prompt=big_prompt, max_tokens=64)]))
    assert trace.step_results[0].tokens_used <= 64


def test_executor_respects_execution_sub_budget():
    ex = Executor(backend=StubBackend())
    steps = [
        _step(sid=f"s{i}", prompt=f"p{i}", max_tokens=500) for i in range(10)
    ]
    # Chain them so they execute in order
    for i in range(1, 10):
        steps[i].depends_on = [steps[i - 1].step_id]
    plan = Plan(
        task_id="t",
        steps=steps,
        strategy=Strategy.ANALYTICAL,
        # Tiny execution budget — should cut execution short.
        budget=Budget(1000, 150, 200, 150, 500),
        gates=Gates(),
    )
    trace = ex.execute_plan(plan)
    assert trace.outcome == TaskOutcome.PARTIAL
    assert trace.error == "execution_budget_exhausted"
    assert len(trace.step_results) < len(steps)


def test_dependency_output_threaded_into_context():
    backend = StubBackend()
    ex = Executor(backend=backend)
    s1 = _step(sid="s1", prompt="first")
    s2 = _step(sid="s2", prompt="second", depends_on=["s1"])
    ex.execute_plan(_plan([s1, s2]))
    # Second call's prompt should include dep marker for s1
    second_call_prompt = backend.calls[1]["prompt"]
    assert "[dep:" in second_call_prompt
    assert "second" in second_call_prompt


def test_working_memory_populated_on_success():
    wm = WorkingMemory()
    ex = Executor(backend=StubBackend())
    ex.execute_plan(_plan([_step()]), working_memory=wm)
    assert "s1" in wm.step_outputs


def test_tool_calls_recorded():
    ex = Executor(backend=StubBackend())
    trace = ex.execute_plan(_plan([_step(tool="retrieval")]))
    assert len(trace.step_results[0].tool_calls) == 1
    assert trace.step_results[0].tool_calls[0]["tool"] == "retrieval"


def test_invalid_plan_cycle_reported():
    a = _step(sid="a", depends_on=["b"])
    b = _step(sid="b", depends_on=["a"])
    ex = Executor(backend=StubBackend())
    trace = ex.execute_plan(_plan([a, b]))
    assert trace.outcome == TaskOutcome.FAILURE
    assert trace.error.startswith("invalid_plan")
