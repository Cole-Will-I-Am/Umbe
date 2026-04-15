from axis.backends.base import BackendResult
from axis.planner import Planner, _extract_json_object
from axis.scheduler import Scheduler
from axis.types import Strategy, Task
from axis.world_model import ActionSchema, WorldModel


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


# ---------------------------------------------------------------------------
# LLM-driven decomposition (§1.2)
# ---------------------------------------------------------------------------
class _ScriptedBackend:
    def __init__(self, *outputs: str):
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def run(self, prompt, max_tokens, tool=None, tool_inventory=None):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        text = self.outputs.pop(0) if self.outputs else ""
        return BackendResult(
            output=text,
            tokens_used=max(1, len(text) // 4),
            success=True,
            entropy=0.0,
        )


_GOOD_JSON_PLAN = """{
  "steps": [
    {"id": "a", "action": "retrieve_context", "tool": "retrieval",
     "strategy": "retrieval_first", "depends_on": []},
    {"id": "b", "action": "reason_about_task", "tool": null,
     "strategy": "analytical", "depends_on": ["a"]},
    {"id": "c", "action": "produce_final_answer", "tool": null,
     "strategy": "analytical", "depends_on": ["b"]}
  ]
}"""


def test_llm_decomposition_parses_json_plan():
    backend = _ScriptedBackend(_GOOD_JSON_PLAN)
    t = Task(input="Explain the CAP theorem", task_class="analysis")
    p, b, g = _prep(t, backend=backend)
    plan = p.plan(t, b, g)
    assert plan.rationale.startswith("llm_decomposition:")
    assert len(plan.steps) == 3
    actions = [s.action for s in plan.steps]
    assert actions == [
        "retrieve_context",
        "reason_about_task",
        "produce_final_answer",
    ]
    # DAG dependencies are resolved to real step ids.
    assert plan.steps[1].depends_on == [plan.steps[0].step_id]
    assert plan.steps[2].depends_on == [plan.steps[1].step_id]


def test_llm_decomposition_handles_code_fenced_response():
    fenced = "```json\n" + _GOOD_JSON_PLAN + "\n```"
    backend = _ScriptedBackend(fenced)
    t = Task(input="x", task_class="analysis")
    p, b, g = _prep(t, backend=backend)
    plan = p.plan(t, b, g)
    assert len(plan.steps) == 3


def test_llm_decomposition_falls_back_on_malformed_json():
    backend = _ScriptedBackend(
        "not json at all",
        "still not json",
        "definitely not json",
    )
    t = Task(input="x", task_class="analysis")
    p, b, g = _prep(t, backend=backend, llm_candidates_k=3)
    plan = p.plan(t, b, g)
    # Should silently fall back to rule-based decomposition.
    assert plan.rationale.startswith("default_decomposition:")
    # All three candidate attempts were made before giving up.
    assert len(backend.calls) == 3


def test_llm_decomposition_falls_back_on_cycle():
    cyclic = """{
      "steps": [
        {"id": "a", "action": "x", "strategy": "analytical",
         "depends_on": ["b"]},
        {"id": "b", "action": "y", "strategy": "analytical",
         "depends_on": ["a"]}
      ]
    }"""
    backend = _ScriptedBackend(cyclic)
    t = Task(input="x", task_class="analysis")
    p, b, g = _prep(t, backend=backend, llm_candidates_k=1)
    plan = p.plan(t, b, g)
    assert plan.rationale.startswith("default_decomposition:")


def test_llm_decomposition_respects_world_model_feasibility():
    infeasible = """{
      "steps": [
        {"id": "a", "action": "run_tool", "strategy": "tool_first",
         "depends_on": []}
      ]
    }"""
    feasible = """{
      "steps": [
        {"id": "a", "action": "prepare", "strategy": "analytical",
         "depends_on": []},
        {"id": "b", "action": "run_tool", "strategy": "tool_first",
         "depends_on": ["a"]}
      ]
    }"""
    wm = WorldModel(
        schemas=[
            ActionSchema(
                action="prepare",
                preconditions=[],
                postconditions_on_success=["ready"],
                estimated_latency_ms=10,
            ),
            ActionSchema(
                action="run_tool",
                preconditions=["ready"],
                postconditions_on_success=["done"],
                estimated_latency_ms=20,
            ),
        ]
    )
    backend = _ScriptedBackend(infeasible, feasible)
    t = Task(input="do the thing", task_class="analysis")
    p, b, g = _prep(t, backend=backend, world_model=wm, llm_candidates_k=2)
    plan = p.plan(t, b, g)
    assert plan.rationale.startswith("llm_decomposition:")
    assert [s.action for s in plan.steps] == ["prepare", "run_tool"]


def test_llm_decomposition_falls_back_when_no_feasible_candidate():
    # Only infeasible candidate — no prerequisite satisfied.
    infeasible = """{
      "steps": [
        {"id": "a", "action": "run_tool", "strategy": "tool_first",
         "depends_on": []}
      ]
    }"""
    wm = WorldModel(
        schemas=[
            ActionSchema(
                action="run_tool",
                preconditions=["ready"],
                postconditions_on_success=["done"],
            )
        ]
    )
    backend = _ScriptedBackend(infeasible, infeasible)
    t = Task(input="x", task_class="analysis")
    p, b, g = _prep(t, backend=backend, world_model=wm, llm_candidates_k=2)
    plan = p.plan(t, b, g)
    assert plan.rationale.startswith("default_decomposition:")


def test_llm_decomposition_skipped_when_procedural_template_matches():
    template = {
        "name": "fixed_template",
        "steps": [
            {"action": "step1", "strategy": "analytical"},
            {"action": "step2", "strategy": "analytical"},
        ],
    }
    backend = _ScriptedBackend(_GOOD_JSON_PLAN)
    t = Task(input="x", task_class="code_debug")
    p, b, g = _prep(
        t, backend=backend, procedural_memory={"code_debug": template}
    )
    plan = p.plan(t, b, g)
    assert plan.rationale.startswith("procedural_template:")
    # Backend was never consulted when a procedural template exists.
    assert backend.calls == []


def test_extract_json_object_handles_strings_with_braces():
    raw = 'prose {"steps": [{"action": "a { fake", "strategy": "analytical"}]} tail'
    extracted = _extract_json_object(raw)
    assert extracted is not None
    import json as _json

    parsed = _json.loads(extracted)
    assert parsed["steps"][0]["action"] == "a { fake"


def test_extract_json_object_returns_none_on_missing_brace():
    assert _extract_json_object("no json here") is None


# ---------------------------------------------------------------------------
# Retriever injection
# ---------------------------------------------------------------------------
def test_retriever_output_is_injected_into_retrieve_context_step():
    calls: list = []

    def _fake_retriever(task_input, task_class):
        calls.append((task_input, task_class))
        return "- fact: the sky is blue\n- fact: water is wet"

    self_model = {
        "capabilities": {"research": {"best_strategy": "retrieval_first"}},
    }
    t = Task(input="What colour is the sky?", task_class="research")
    p, b, g = _prep(t, self_model=self_model, retriever=_fake_retriever)
    plan = p.plan(t, b, g)

    retrieve_step = next(s for s in plan.steps if s.action == "retrieve_context")
    assert "[retrieved_context]" in retrieve_step.prompt
    assert "the sky is blue" in retrieve_step.prompt
    assert calls == [("What colour is the sky?", "research")]


def test_retriever_exception_does_not_break_planning():
    def _boom(_i, _c):
        raise RuntimeError("retriever exploded")

    self_model = {
        "capabilities": {"research": {"best_strategy": "retrieval_first"}},
    }
    t = Task(input="x", task_class="research")
    p, b, g = _prep(t, self_model=self_model, retriever=_boom)
    plan = p.plan(t, b, g)
    retrieve_step = next(s for s in plan.steps if s.action == "retrieve_context")
    # No retrieved_context block means we fell back to the plain prompt.
    assert "[retrieved_context]" not in retrieve_step.prompt


def test_retriever_empty_result_leaves_prompt_unwrapped():
    def _empty(task_input, task_class):
        return ""

    self_model = {
        "capabilities": {"research": {"best_strategy": "retrieval_first"}},
    }
    t = Task(input="x", task_class="research")
    p, b, g = _prep(t, self_model=self_model, retriever=_empty)
    plan = p.plan(t, b, g)
    retrieve_step = next(s for s in plan.steps if s.action == "retrieve_context")
    assert "[retrieved_context]" not in retrieve_step.prompt
