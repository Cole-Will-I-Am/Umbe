from axis.reference_frame import (
    ArchitectureAwareness,
    ComputationPosition,
    GradientSignals,
    InteractionModel,
    KnowledgeTopology,
    ResourceHorizon,
)


def test_computation_position_stops_when_budget_exhausted():
    cp = ComputationPosition(tokens_used=1000, tokens_remaining=0)
    assert cp.should_force_stop(marginal_gain_estimate=0.9)


def test_computation_position_stops_on_diminishing_returns():
    cp = ComputationPosition(tokens_used=900, tokens_remaining=100)
    assert cp.should_force_stop(marginal_gain_estimate=0.05)
    # Still strong gain → keep going
    assert not cp.should_force_stop(marginal_gain_estimate=0.5)


def test_knowledge_topology_retrieval_depth():
    strong = KnowledgeTopology(
        domain="chat", domain_confidence=0.9, frontier_distance=0.1
    )
    assert strong.retrieval_depth() == "none"

    unknown = KnowledgeTopology(
        domain="novel", domain_confidence=0.4, frontier_distance=0.9
    )
    assert unknown.retrieval_depth() == "broad_sweep"

    middling = KnowledgeTopology(
        domain="analysis", domain_confidence=0.7, frontier_distance=0.4
    )
    assert middling.retrieval_depth() == "targeted"


def test_knowledge_topology_verification_depth():
    weak = KnowledgeTopology(
        domain="legal", domain_confidence=0.4, frontier_distance=0.8
    )
    assert weak.verification_depth() == "full"
    strong = KnowledgeTopology(
        domain="chat", domain_confidence=0.95, frontier_distance=0.1
    )
    assert strong.verification_depth() == "spot_check"


def test_architecture_awareness_plan_feasibility():
    aa = ArchitectureAwareness(
        context_window_tokens=100_000,
        context_tokens_used=30_000,
        tools_available=["retrieval", "code_exec"],
    )
    assert aa.plan_is_feasible(40_000, ["retrieval"])
    # Over the window
    assert not aa.plan_is_feasible(80_000, ["retrieval"])
    # Missing tool
    assert not aa.plan_is_feasible(1000, ["sql"])


def test_resource_horizon_must_prioritise():
    rh = ResourceHorizon(
        remaining_task_tokens=200, session_tokens_remaining=10_000
    )
    assert rh.must_prioritise()
    rh2 = ResourceHorizon(
        remaining_task_tokens=5000,
        session_tokens_remaining=50_000,
        wall_clock_remaining_ms=500,
    )
    assert rh2.must_prioritise()


def test_interaction_model_response_posture():
    assert (
        InteractionModel(stakes="critical").response_posture() == "cautious"
    )
    assert (
        InteractionModel(stakes="low", collaborator_expertise="novice").response_posture()
        == "thorough"
    )
    assert (
        InteractionModel(stakes="medium", channel="voice").response_posture()
        == "concise"
    )


def test_gradient_signals_gate_decisions():
    confident = GradientSignals(
        confidence=0.9, novelty=0.1, coherence=0.95, friction=0
    )
    assert not confident.should_retrieve()
    assert not confident.should_verify()
    assert not confident.should_replan()

    struggling = GradientSignals(
        confidence=0.3, novelty=0.7, coherence=0.5, friction=5
    )
    assert struggling.should_retrieve()
    assert struggling.should_verify()
    assert struggling.should_replan()
    assert struggling.should_escalate()
