from axis.backends.stub import StubBackend
from axis.executor import Executor
from axis.runtime import AxisRuntime
from axis.types import Stakes, Task, TaskOutcome


def test_runtime_runs_end_to_end():
    rt = AxisRuntime(executor=Executor(backend=StubBackend()))
    t = Task(input="Analyze this document", task_class="analysis", stakes=Stakes.MEDIUM)
    trace = rt.run(t)
    assert trace.outcome == TaskOutcome.SUCCESS
    assert len(trace.step_results) > 0
    assert trace.total_tokens > 0
    assert trace.latency_ms() >= 0


def test_runtime_replans_on_failure():
    """Every default-decomposition plan contains a 'reason_about_task' step;
    failing that trigger forces the runtime to hit the replan path."""
    backend = StubBackend(fail_on={"reason_about_task"})
    rt = AxisRuntime(executor=Executor(backend=backend), max_replans=2)
    trace = rt.run(Task(input="Solve this", task_class="general"))
    assert trace.replanning_events >= 1
    # With this stub, retries still contain the failure trigger, so the
    # final outcome stays FAILURE. The important assertion is that replan
    # was attempted rather than improvised.
    assert trace.outcome == TaskOutcome.FAILURE


def test_runtime_gives_up_after_max_replans():
    backend = StubBackend(fail_on={"reason_about_task"})
    rt = AxisRuntime(executor=Executor(backend=backend), max_replans=3)
    trace = rt.run(Task(input="x"))
    assert trace.replanning_events == 3


def test_runtime_uses_scheduler_budget():
    rt = AxisRuntime(executor=Executor(backend=StubBackend()))
    t = Task(input="cheap", task_class="general", stakes=Stakes.LOW, complexity=0.0)
    trace = rt.run(t)
    # Low stakes + zero complexity gives the smallest budget tier.
    assert trace.total_tokens <= 4000


def test_runtime_high_stakes_gets_more_budget_headroom():
    rt = AxisRuntime(executor=Executor(backend=StubBackend()))
    low = rt.run(Task(input="x", stakes=Stakes.LOW, complexity=0.5))
    high = rt.run(Task(input="x", stakes=Stakes.HIGH, complexity=0.5))
    # With the stub backend, more budget means more steps may fit — at minimum
    # the high-stakes run must not use fewer tokens than the low-stakes one.
    assert high.total_tokens >= low.total_tokens
