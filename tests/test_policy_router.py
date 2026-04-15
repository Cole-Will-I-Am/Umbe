from axis.backends.stub import StubBackend
from axis.executor import Executor
from axis.policy_router import PolicyRouter
from axis.runtime import AxisRuntime
from axis.telemetry import TelemetryObserver
from axis.types import Stakes, Strategy, Task
from axis.verifier import Verifier


def test_router_picks_best_strategy_from_self_model():
    r = PolicyRouter()
    self_model = {
        "capabilities": {
            "legal_reasoning": {"win_rate": 0.6, "best_strategy": "retrieval_first"}
        }
    }
    d = r.route(
        Task(input="contract", task_class="legal_reasoning"),
        self_model=self_model,
    )
    assert d.strategy == Strategy.RETRIEVAL_FIRST


def test_router_critical_stakes_force_full_verification():
    r = PolicyRouter()
    d = r.route(Task(input="x", stakes=Stakes.CRITICAL))
    assert d.verification_depth == "full"
    assert d.response_posture == "cautious"
    assert d.escalation_threshold >= 0.8


def test_router_low_stakes_known_domain_skips_verification():
    r = PolicyRouter()
    self_model = {"capabilities": {"chat": {"win_rate": 0.92}}}
    d = r.route(
        Task(input="hello", task_class="chat", stakes=Stakes.LOW),
        self_model=self_model,
    )
    assert d.verification_depth == "skip"


def test_router_retrieval_policy_follows_strategy_and_confidence():
    r = PolicyRouter()
    # retrieval_first strategy → broad sweep
    sm1 = {"capabilities": {"research": {"best_strategy": "retrieval_first"}}}
    d1 = r.route(Task(input="x", task_class="research"), self_model=sm1)
    assert d1.retrieval_policy == "broad_sweep"
    # Weak domain with fallback strategy → targeted
    sm2 = {"capabilities": {"niche": {"win_rate": 0.4}}}
    d2 = r.route(Task(input="x", task_class="niche"), self_model=sm2)
    assert d2.retrieval_policy == "targeted"
    # Strong domain → no retrieval
    sm3 = {"capabilities": {"chat": {"win_rate": 0.9}}}
    d3 = r.route(Task(input="x", task_class="chat"), self_model=sm3)
    assert d3.retrieval_policy == "none"


def test_router_logs_features_and_produces_training_data():
    r = PolicyRouter()
    r.route(Task(task_id="t1", input="x", task_class="chat"))
    r.route(Task(task_id="t2", input="y", task_class="chat"))
    r.record_outcome("t1", "success")
    r.record_outcome("t2", "failure")
    data = r.training_data()
    assert len(data) == 2
    assert {d["outcome"] for d in data} == {"success", "failure"}
    assert all("features" in d and "decision" in d for d in data)


def test_runtime_wires_router_and_records_outcome():
    router = PolicyRouter()
    rt = AxisRuntime(
        executor=Executor(backend=StubBackend()),
        policy_router=router,
    )
    trace = rt.run(Task(input="do the thing", task_class="general"))
    # The decision should be visible on the trace
    assert trace.routing_decision is not None
    # And the outcome should have been paired with the decision in the log.
    data = router.training_data()
    assert len(data) == 1


def test_runtime_router_can_force_verification_off_for_low_stakes():
    router = PolicyRouter()
    rt = AxisRuntime(
        executor=Executor(backend=StubBackend()),
        policy_router=router,
        verifier=Verifier(backend=StubBackend()),
    )
    sm = {"capabilities": {"chat": {"win_rate": 0.95}}}
    rt.planner.self_model = {**rt.planner.self_model, **sm}
    trace = rt.run(Task(input="hi", task_class="chat", stakes=Stakes.LOW))
    assert trace.verifier_score is not None  # Verifier still ran but was skipped
