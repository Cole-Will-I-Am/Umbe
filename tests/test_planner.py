from axis.planner import Planner
from axis.scheduler import Scheduler
from axis.types import Strategy, Task


def _prep(task: Task, **planner_kwargs):
    s = Scheduler()
    p = Planner(**planner_kwargs)
    budget = s.allocate(task)
    gates = s.gates(task)
    return p, budget, gates


def test_plan_has_steps_and_task_id():
    t = Task(input="Find the bug", task_class="code_debug")
    p, b, g = _prep(t)
    plan = p.plan(t, b, g)
    assert len(plan.steps) > 0
    assert plan.task_id == t.task_id


def test_plan_steps_are_topologically_ordered():
    t = Task(input="Analyze document", task_class="analysis")
    p, b, g = _prep(t)
    plan = p.plan(t, b, g)
    order = plan.topological_order()
    seen: set[str] = set()
    for step in order:
        for dep in step.depends_on:
            assert dep in seen, "dependency visited before dependent"
        seen.add(step.step_id)


def test_cycle_detection_raises():
    from axis.types import Plan, PlanStep, Budget, Gates

    a = PlanStep(step_id="a", action="x", strategy=Strategy.ANALYTICAL, prompt="")
    b = PlanStep(
        step_id="b", action="y", strategy=Strategy.ANALYTICAL, prompt="", depends_on=["a"]
    )
    a.depends_on = ["b"]  # cycle
    plan = Plan(
        task_id="t",
        steps=[a, b],
        strategy=Strategy.ANALYTICAL,
        budget=Budget(1000, 150, 550, 150, 150),
        gates=Gates(),
    )
    import pytest

    with pytest.raises(ValueError):
        plan.topological_order()


def test_self_model_best_strategy_wins():
    self_model = {
        "capabilities": {"legal_reasoning": {"best_strategy": "retrieval_first"}},
        "strategy_effectiveness": {
            "tool_first": {"win_rate_when_selected": 0.99},
        },
    }
    t = Task(input="Summarize contract", task_class="legal_reasoning")
    p, b, g = _prep(t, self_model=self_model)
    plan = p.plan(t, b, g)
    assert plan.strategy == Strategy.RETRIEVAL_FIRST


def test_retrieval_first_strategy_inserts_retrieval_step():
    self_model = {
        "capabilities": {"research": {"best_strategy": "retrieval_first"}},
    }
    t = Task(input="What's new in ML?", task_class="research")
    p, b, g = _prep(t, self_model=self_model)
    plan = p.plan(t, b, g)
    actions = [s.action for s in plan.steps]
    assert "retrieve_context" in actions
    assert actions[-1] == "produce_final_answer"


def test_tool_first_strategy_inserts_tool_step():
    self_model = {
        "capabilities": {"math_symbolic": {"best_strategy": "tool_first"}},
    }
    t = Task(input="integrate x^2", task_class="math_symbolic")
    p, b, g = _prep(t, self_model=self_model)
    plan = p.plan(t, b, g)
    actions = [s.action for s in plan.steps]
    assert "invoke_primary_tool" in actions


def test_replan_swaps_failed_step_and_changes_strategy():
    t = Task(input="debug this", task_class="code_debug")
    p, b, g = _prep(t)
    plan = p.plan(t, b, g)
    original_ids = {s.step_id for s in plan.steps}
    failed_id = plan.steps[-1].step_id

    new_plan = p.replan(t, plan, failed_id, error="boom")

    assert new_plan.strategy != plan.strategy
    assert len(new_plan.steps) == len(plan.steps)
    new_step = next(s for s in new_plan.steps if s.step_id not in original_ids)
    assert new_step.action.endswith("_retry")
    assert "replan" in new_plan.rationale


def test_procedural_template_used_when_available():
    template = {
        "name": "debug_python_with_tests",
        "steps": [
            {
                "action": "retrieve_error_context",
                "tool": "code_search",
                "strategy": "retrieval_first",
            },
            {"action": "hypothesize_root_cause", "strategy": "analytical"},
            {"action": "generate_fix", "strategy": "analytical"},
            {"action": "run_tests", "tool": "test_runner", "strategy": "tool_first"},
        ],
    }
    t = Task(input="buggy", task_class="code_debug")
    p, b, g = _prep(t, procedural_memory={"code_debug": template})
    plan = p.plan(t, b, g)
    assert plan.rationale.startswith("procedural_template:")
    assert len(plan.steps) == 4
    assert plan.steps[0].action == "retrieve_error_context"
    assert plan.steps[0].tool == "code_search"
    assert plan.steps[-1].tool == "test_runner"


def test_per_step_max_tokens_positive():
    t = Task(input="x", task_class="general")
    p, b, g = _prep(t)
    plan = p.plan(t, b, g)
    for s in plan.steps:
        assert s.max_tokens >= 256
