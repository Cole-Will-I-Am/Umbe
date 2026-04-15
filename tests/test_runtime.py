from axis.backends.stub import StubBackend
from axis.executor import Executor
from axis.memory import Episode, EpisodicMemory, MemoryManager
from axis.memory.embedding import embed
from axis.planner import Planner
from axis.runtime import AxisRuntime
from axis.telemetry import StructuredSink, TelemetryObserver
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


def test_runtime_emits_telemetry_to_observer():
    obs = TelemetryObserver()
    rt = AxisRuntime(executor=Executor(backend=StubBackend()), observer=obs)
    rt.run(Task(input="analyze this", task_class="analysis"))
    metrics = obs.metrics()
    assert metrics["tasks_total"] == 1
    assert "analysis" in metrics["win_rate_by_class"]
    assert metrics["strategy_bias"]  # at least one strategy recorded


def test_runtime_telemetry_writes_to_structured_sink(tmp_path):
    sink = StructuredSink(tmp_path / "telemetry.jsonl")
    obs = TelemetryObserver(sink=sink)
    rt = AxisRuntime(executor=Executor(backend=StubBackend()), observer=obs)
    rt.run(Task(input="x", task_class="general"))
    rt.run(Task(input="y", task_class="general"))
    records = sink.read_all()
    assert len(records) == 2
    assert all(r["task_class"] == "general" for r in records)


def test_runtime_self_model_refresh_hook_runs():
    """Only the Observer may write the Self-Model (§6). After enough tasks
    the runtime should pull a fresh snapshot into the Planner."""
    obs = TelemetryObserver(self_model_refresh_every=2)
    rt = AxisRuntime(
        executor=Executor(backend=StubBackend()),
        observer=obs,
        auto_refresh_self_model=True,
    )
    before = dict(rt.planner.self_model)
    rt.run(Task(input="x", task_class="chat"))
    rt.run(Task(input="y", task_class="chat"))
    after = rt.planner.self_model
    # Capabilities dict should now reflect measured telemetry
    assert "chat" in after.get("capabilities", {})
    assert after["capabilities"]["chat"]["sample_size"] >= 1
    # The bootstrap fields not replaced by the snapshot should still be there
    assert "architecture" in after
    assert after is not before


def test_runtime_auto_wires_memory_retriever_into_planner():
    """When a MemoryManager is configured, the Runtime should install a
    retriever on the Planner so retrieve_context steps get real
    episodic + semantic hits injected."""
    memory = MemoryManager()
    # Seed an episode that should show up on a similar query.
    memory.episodic.add(
        Episode(
            episode_id="e1",
            timestamp=0.0,
            task_hash="h",
            task_class="research",
            input_summary="Previous research: transformer attention scaling laws",
            strategy_used="analytical",
            tools_invoked=[],
            outcome="success",
            embedding=embed("transformer attention scaling laws"),
        )
    )
    memory.semantic.upsert(
        type="fact",
        content="Attention is all you need (Vaswani et al. 2017)",
        confidence=0.9,
    )

    planner = Planner(
        self_model={
            "capabilities": {"research": {"best_strategy": "retrieval_first"}}
        }
    )
    assert planner.retriever is None

    rt = AxisRuntime(
        executor=Executor(backend=StubBackend()),
        planner=planner,
        memory=memory,
    )
    # Retriever installed.
    assert rt.planner.retriever is not None

    t = Task(
        input="Explain transformer attention scaling laws",
        task_class="research",
    )
    trace = rt.run(t)
    assert trace.outcome == TaskOutcome.SUCCESS

    # The stub backend echoes the prompt prefix back; we can search the
    # captured step outputs for evidence the retrieved context reached
    # the executor.
    all_output = "".join(sr.output for sr in trace.step_results)
    assert "retrieved_context" in all_output


def test_runtime_does_not_override_existing_retriever():
    sentinel = lambda inp, cls: "sentinel-output"
    planner = Planner(retriever=sentinel)
    rt = AxisRuntime(
        executor=Executor(backend=StubBackend()),
        planner=planner,
        memory=MemoryManager(),
    )
    assert rt.planner.retriever is sentinel


def test_runtime_without_memory_leaves_retriever_unset():
    planner = Planner()
    rt = AxisRuntime(executor=Executor(backend=StubBackend()), planner=planner)
    assert rt.planner.retriever is None


def test_runtime_captures_failure_in_telemetry():
    backend = StubBackend(fail_on={"reason_about_task"})
    obs = TelemetryObserver()
    rt = AxisRuntime(
        executor=Executor(backend=backend),
        observer=obs,
        max_replans=1,
    )
    rt.run(Task(input="x", task_class="general"))
    metrics = obs.metrics()
    assert metrics["win_rate_by_class"]["general"] == 0.0
    assert metrics["replanning_rate"] == 1.0
    # Error string should show up in failure_mode_frequency
    assert any(
        "stub_triggered_failure" in k for k in metrics["failure_mode_frequency"]
    )
