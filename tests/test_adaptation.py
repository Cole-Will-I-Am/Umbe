import time

from axis.adaptation import (
    AdaptationConfig,
    AdaptationManager,
    ChangeClass,
    Proposal,
    _classify_root_cause,
)
from axis.eval_harness import EvalCase, EvalHarness
from axis.memory import Episode, MemoryManager
from axis.memory.embedding import embed
from axis.telemetry import TelemetryObserver
from axis.types import (
    ExecutionTrace,
    Strategy,
    Task,
    TaskOutcome,
)


def _ingest_failures(obs: TelemetryObserver, mm: MemoryManager, task_class: str, n: int, error: str = "stale_retrieval"):
    for i in range(n):
        trace = ExecutionTrace(
            task_id=f"f{i}",
            task_class=task_class,
            strategy=Strategy.ANALYTICAL,
            execution_tokens=100,
            total_tokens=100,
            outcome=TaskOutcome.FAILURE,
            error=error,
        )
        task = Task(task_id=f"f{i}", input=f"case {i}", task_class=task_class)
        obs.record_trace(trace, task)
        mm.record_episode(trace, task)


def test_classify_root_cause_mapping():
    ep = Episode(
        episode_id="e",
        timestamp=0.0,
        task_hash="h",
        task_class="c",
        input_summary="",
        strategy_used=None,
        tools_invoked=[],
        outcome="failure",
        outcome_detail="stale_retrieval hit",
    )
    assert _classify_root_cause(ep, "stale") == "retrieval_failure"
    ep.outcome_detail = "tool_timeout"
    assert _classify_root_cause(ep, "") == "tool_selection_error"
    ep.outcome_detail = "over_confident claim"
    assert _classify_root_cause(ep, "") == "confidence_miscalibration"
    ep.outcome_detail = "something weird"
    assert _classify_root_cause(ep, "") == "strategy_mismatch"


def test_triggers_activate_on_failure_spike():
    obs = TelemetryObserver()
    mm = MemoryManager()
    mgr = AdaptationManager(obs, mm)
    _ingest_failures(obs, mm, "legal_reasoning", n=12)
    triggers = mgr.check_triggers()
    assert any(t.startswith("failure_spike") for t in triggers)


def test_diagnose_returns_none_when_too_few_failures():
    obs = TelemetryObserver()
    mm = MemoryManager()
    mgr = AdaptationManager(obs, mm, config=AdaptationConfig(failure_cluster_min=10))
    _ingest_failures(obs, mm, "code_debug", n=3)
    assert mgr.diagnose("code_debug") is None


def test_diagnose_picks_dominant_root_cause():
    obs = TelemetryObserver()
    mm = MemoryManager()
    mgr = AdaptationManager(obs, mm, config=AdaptationConfig(failure_cluster_min=10))
    _ingest_failures(obs, mm, "legal_reasoning", n=12, error="stale_retrieval")
    diag = mgr.diagnose("legal_reasoning")
    assert diag is not None
    assert diag["root_cause"] == "retrieval_failure"
    assert diag["failure_count"] == 12


def test_propose_generates_scoped_policy_change():
    obs = TelemetryObserver()
    mm = MemoryManager()
    mgr = AdaptationManager(obs, mm, config=AdaptationConfig(failure_cluster_min=10))
    _ingest_failures(obs, mm, "legal_reasoning", n=12, error="stale_retrieval")
    diag = mgr.diagnose("legal_reasoning")
    proposal = mgr.propose(diag)
    assert proposal is not None
    assert proposal.change_class == ChangeClass.POLICY
    assert proposal.target_task_class == "legal_reasoning"
    assert "legal_reasoning" in proposal.affected_task_classes
    assert proposal.rollback_trigger["threshold"] > 0
    assert proposal.counterfactual_improvement > 0


def test_propose_tool_error_generates_class_2_procedural():
    obs = TelemetryObserver()
    mm = MemoryManager()
    mgr = AdaptationManager(obs, mm, config=AdaptationConfig(failure_cluster_min=10))
    _ingest_failures(obs, mm, "code_debug", n=12, error="tool_timeout")
    diag = mgr.diagnose("code_debug")
    prop = mgr.propose(diag)
    assert prop is not None
    assert prop.change_class == ChangeClass.MEMORY


def test_deploy_records_deployment_for_rollback_tracking():
    obs = TelemetryObserver()
    mm = MemoryManager()
    mgr = AdaptationManager(obs, mm)

    applied = {}

    def apply_fn(payload):
        applied.update(payload)

    proposal = Proposal(
        proposal_id="p",
        change_class=ChangeClass.POLICY,
        target_task_class="chat",
        root_cause="strategy_mismatch",
        description="desc",
        counterfactual_improvement=0.1,
        affected_task_classes=["chat"],
        rollback_trigger={"metric": "win_rate", "task_class": "chat", "threshold": 0.05, "window": 50},
        payload={"policy": "best_strategy", "set": "retrieval_first"},
    )
    dep = mgr.deploy(proposal, apply_fn)
    assert applied["set"] == "retrieval_first"
    assert dep in mgr.deployments
    assert dep.rolled_back is False


def test_rollback_fires_on_win_rate_drop():
    obs = TelemetryObserver()
    mm = MemoryManager()
    mgr = AdaptationManager(obs, mm)

    # Warm up baseline win rate of 1.0 for "chat"
    for i in range(5):
        trace = ExecutionTrace(
            task_id=f"t{i}",
            task_class="chat",
            strategy=Strategy.ANALYTICAL,
            execution_tokens=10,
            total_tokens=10,
            outcome=TaskOutcome.SUCCESS,
        )
        obs.record_trace(trace, Task(task_id=f"t{i}", input="x", task_class="chat"))

    proposal = Proposal(
        proposal_id="p",
        change_class=ChangeClass.POLICY,
        target_task_class="chat",
        root_cause="strategy_mismatch",
        description="desc",
        counterfactual_improvement=0.1,
        affected_task_classes=["chat"],
        rollback_trigger={"metric": "win_rate", "task_class": "chat", "threshold": 0.05, "window": 50},
        payload={},
    )
    mgr.deploy(proposal, apply_fn=lambda p: None)

    # Now win rate craters
    for i in range(10):
        trace = ExecutionTrace(
            task_id=f"bad{i}",
            task_class="chat",
            strategy=Strategy.ANALYTICAL,
            execution_tokens=10,
            total_tokens=10,
            outcome=TaskOutcome.FAILURE,
            error="x",
        )
        obs.record_trace(trace, Task(task_id=f"bad{i}", input="x", task_class="chat"))

    rolled = mgr.check_rollbacks()
    assert len(rolled) == 1
    assert rolled[0].rolled_back
