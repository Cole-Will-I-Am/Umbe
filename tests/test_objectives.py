import pytest

from axis.objectives import DEFAULT_DOMAIN_OVERRIDES, ObjectiveStack
from axis.types import Stakes, Task


def test_default_weights_sum_to_one_after_normalisation():
    o = ObjectiveStack()
    w = o.weights_for(Task(input="x", task_class="general"))
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_domain_override_shifts_weights():
    o = ObjectiveStack()
    medical = o.weights_for(Task(input="x", task_class="medical"))
    general = o.weights_for(Task(input="x", task_class="general"))
    assert medical["truthfulness"] > general["truthfulness"]
    assert medical["originality"] == 0.0  # override forces to 0


def test_critical_stakes_zero_latency_weight():
    o = ObjectiveStack()
    w = o.weights_for(Task(input="x", stakes=Stakes.CRITICAL))
    assert w["latency"] == 0.0


def test_high_stakes_boosts_truthfulness():
    o = ObjectiveStack()
    low = o.weights_for(Task(input="x", stakes=Stakes.LOW))
    high = o.weights_for(Task(input="x", stakes=Stakes.HIGH))
    assert high["truthfulness"] > low["truthfulness"]


def test_conflict_resolution_picks_highest_weighted_score():
    o = ObjectiveStack()
    task = Task(input="x", task_class="general")
    candidates = {
        "approach_a": {
            "truthfulness": 0.9,
            "task_success": 0.9,
            "calibration": 0.9,
            "helpfulness": 0.8,
            "efficiency": 0.7,
            "latency": 0.5,
            "robustness": 0.6,
            "originality": 0.2,
            "learning_value": 0.1,
        },
        "approach_b": {
            "truthfulness": 0.3,
            "task_success": 0.5,
            "calibration": 0.4,
            "helpfulness": 0.4,
            "efficiency": 0.9,
            "latency": 0.9,
            "robustness": 0.3,
            "originality": 0.6,
            "learning_value": 0.5,
        },
    }
    result = o.resolve(task, candidates)
    assert result.chosen == "approach_a"
    assert result.score > 0


def test_conflict_resolution_uses_robustness_tiebreaker():
    o = ObjectiveStack()
    task = Task(input="x")
    # Two candidates scored within 5% of each other
    candidates = {
        "risky": {"truthfulness": 0.80, "task_success": 0.80, "calibration": 0.80},
        "safe": {"truthfulness": 0.78, "task_success": 0.78, "calibration": 0.78},
    }
    variance = {"risky": 0.5, "safe": 0.05}
    result = o.resolve(task, candidates, variance=variance)
    assert result.chosen == "safe"
    assert result.used_tiebreaker


def test_conflict_resolution_drops_candidates_failing_hard_constraint():
    o = ObjectiveStack()
    task = Task(input="x")
    candidates = {
        "unsafe": {"task_success": 0.99},
        "safe": {"task_success": 0.5},
    }
    result = o.resolve(
        task,
        candidates,
        hard_constraint_fn=lambda name: name != "unsafe",
    )
    assert result.chosen == "safe"


def test_conflict_resolution_exploration_flag_on_novel_tasks():
    o = ObjectiveStack()
    result = o.resolve(
        Task(input="x"),
        {"a": {"task_success": 0.9}, "b": {"task_success": 0.5}},
        novelty=0.9,
    )
    assert result.exploration_allocated


def test_resolve_raises_when_all_candidates_fail_constraint():
    o = ObjectiveStack()
    with pytest.raises(ValueError):
        o.resolve(
            Task(input="x"),
            {"a": {"task_success": 0.9}},
            hard_constraint_fn=lambda _: False,
        )
