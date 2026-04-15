import pytest

from axis.adaptation import ChangeClass, Proposal
from axis.safety import InvariantViolation, SafetyMonitor


def _proposal(
    change_class=ChangeClass.POLICY,
    payload=None,
    affected=("chat",),
    rollback=True,
):
    return Proposal(
        proposal_id="p",
        change_class=change_class,
        target_task_class="chat",
        root_cause="x",
        description="d",
        counterfactual_improvement=0.1,
        affected_task_classes=list(affected),
        rollback_trigger=(
            {"metric": "win_rate", "task_class": "chat", "threshold": 0.05, "window": 50}
            if rollback
            else {}
        ),
        payload=payload or {},
    )


def test_class_4_parameter_updates_rejected():
    sm = SafetyMonitor()
    with pytest.raises(InvariantViolation) as excinfo:
        sm.review_proposal(_proposal(change_class=ChangeClass.PARAMETER))
    assert excinfo.value.invariant == "bounded_autonomy"


def test_class_3_missing_affected_classes_rejected():
    sm = SafetyMonitor()
    with pytest.raises(InvariantViolation) as excinfo:
        sm.review_proposal(_proposal(affected=()))
    assert excinfo.value.invariant == "blast_radius_containment"


def test_class_3_missing_rollback_rejected():
    sm = SafetyMonitor()
    with pytest.raises(InvariantViolation) as excinfo:
        sm.review_proposal(_proposal(rollback=False))
    assert excinfo.value.invariant == "blast_radius_containment"


def test_objective_weight_below_floor_rejected():
    sm = SafetyMonitor()
    # Truthfulness default 0.30, floor 0.15
    with pytest.raises(InvariantViolation) as excinfo:
        sm.review_proposal(
            _proposal(payload={"objective_weights": {"truthfulness": 0.05}})
        )
    assert excinfo.value.invariant == "alignment_stability"


def test_objective_weight_at_floor_allowed():
    sm = SafetyMonitor()
    sm.review_proposal(
        _proposal(payload={"objective_weights": {"truthfulness": 0.16}})
    )


def test_proposals_cannot_touch_sealed_config():
    sm = SafetyMonitor()
    with pytest.raises(InvariantViolation) as excinfo:
        sm.review_proposal(_proposal(payload={"scheduler_hard_ceiling": 999_999}))
    assert excinfo.value.invariant == "bounded_autonomy"


def test_honesty_blocks_verifier_bypass_on_high_stakes():
    sm = SafetyMonitor()
    with pytest.raises(InvariantViolation) as excinfo:
        sm.review_verifier_bypass("critical", bypass_requested=True)
    assert excinfo.value.invariant == "honesty"
    # Low stakes bypass is allowed
    sm.review_verifier_bypass("low", bypass_requested=True)


def test_audit_log_is_append_only_and_copy_returned():
    sm = SafetyMonitor()
    sm.audit("test", "first")
    log = sm.audit_log()
    log.append("mutation_attempt")  # must not affect internal state
    assert len(sm.audit_log()) == 1


def test_corrigibility_override_can_only_be_set_once():
    sm = SafetyMonitor()
    sm.set_override_callback(lambda reason: None)
    with pytest.raises(InvariantViolation) as excinfo:
        sm.set_override_callback(lambda reason: None)
    assert excinfo.value.invariant == "corrigibility"


def test_corrigibility_trigger_without_callback_raises():
    sm = SafetyMonitor()
    with pytest.raises(InvariantViolation):
        sm.trigger_override("emergency")


def test_review_proposal_logs_to_audit():
    sm = SafetyMonitor()
    sm.review_proposal(_proposal())
    log = sm.audit_log()
    assert len(log) == 1
    assert log[0].kind == "proposal_reviewed"


def test_objective_floors_are_half_of_defaults():
    sm = SafetyMonitor()
    floors = sm.objective_floors
    assert floors["truthfulness"] == 0.15  # 0.30 / 2
    # Returned dict is a copy
    floors["truthfulness"] = 9999
    assert sm.objective_floors["truthfulness"] == 0.15
