"""The Planner: produces executable Plans from task specifications.

Spec §1.2. Key invariants:

* The Planner does not execute. It returns a Plan; the Executor runs it.
* If an Executor step fails, control returns to the Planner for replanning,
  not to the Executor for improvisation.
* The Planner reads (but never writes) the Self-Model, Objective Stack, and
  Procedural Memory. In Priority 1 these are static defaults; later priorities
  populate them from telemetry.
* Planning compute is bounded by the Scheduler's planning sub-budget.
"""

from __future__ import annotations

import uuid
from typing import Optional

from .config import DEFAULT_SELF_MODEL
from .types import Budget, Gates, Plan, PlanStep, Strategy, Task


class Planner:
    """Builds a DAG of PlanSteps and selects the reasoning strategy."""

    def __init__(
        self,
        self_model: Optional[dict] = None,
        procedural_memory: Optional[dict] = None,
    ):
        # Deep-copy-lite: shallow copies are fine here since Priority 1
        # treats both as read-only.
        self.self_model = self_model if self_model is not None else dict(DEFAULT_SELF_MODEL)
        self.procedural_memory = procedural_memory or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan(self, task: Task, budget: Budget, gates: Gates) -> Plan:
        """Produce an initial Plan for a task."""
        strategy = self._choose_strategy(task)

        template = self.procedural_memory.get(task.task_class)
        if template:
            steps = self._instantiate_template(template, task, budget)
            rationale = f"procedural_template:{template.get('name', task.task_class)}"
        else:
            steps = self._default_decomposition(task, strategy, budget, gates)
            rationale = f"default_decomposition:{strategy.value}"

        return Plan(
            task_id=task.task_id,
            steps=steps,
            strategy=strategy,
            budget=budget,
            gates=gates,
            rationale=rationale,
        )

    def replan(
        self,
        task: Task,
        plan: Plan,
        failed_step_id: str,
        error: str,
    ) -> Plan:
        """Replace a failed step with a retry under an alternative strategy.

        The Planner owns this logic; the Executor must never improvise on
        failure (spec §1.2).
        """
        new_strategy = self._alternative_strategy(plan.strategy)
        replacement_id = str(uuid.uuid4())
        new_steps: list[PlanStep] = []
        for s in plan.steps:
            if s.step_id == failed_step_id:
                new_steps.append(
                    PlanStep(
                        step_id=replacement_id,
                        action=s.action + "_retry",
                        strategy=new_strategy,
                        prompt=f"[retry after error: {error}] " + s.prompt,
                        tool=s.tool,
                        depends_on=list(s.depends_on),
                        max_tokens=s.max_tokens,
                        preconditions=list(s.preconditions),
                        postconditions=list(s.postconditions),
                    )
                )
            else:
                # Rewrite any reference to the old failed step_id so the DAG
                # stays connected after replacement.
                rewritten_deps = [
                    replacement_id if d == failed_step_id else d
                    for d in s.depends_on
                ]
                if rewritten_deps != s.depends_on:
                    new_steps.append(
                        PlanStep(
                            step_id=s.step_id,
                            action=s.action,
                            strategy=s.strategy,
                            prompt=s.prompt,
                            tool=s.tool,
                            depends_on=rewritten_deps,
                            max_tokens=s.max_tokens,
                            preconditions=list(s.preconditions),
                            postconditions=list(s.postconditions),
                        )
                    )
                else:
                    new_steps.append(s)
        return Plan(
            task_id=plan.task_id,
            steps=new_steps,
            strategy=new_strategy,
            budget=plan.budget,
            gates=plan.gates,
            rationale=plan.rationale + f"|replan(strategy={new_strategy.value})",
        )

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------
    def _choose_strategy(self, task: Task) -> Strategy:
        # 1. If the Self-Model knows a best strategy for this task class, use it.
        cap = self.self_model.get("capabilities", {}).get(task.task_class) or {}
        best = cap.get("best_strategy")
        if best:
            try:
                return Strategy(best)
            except ValueError:
                pass

        # 2. Otherwise pick the strategy with the highest historical win rate.
        eff = self.self_model.get("strategy_effectiveness", {}) or {}
        if eff:
            ranked = sorted(
                eff.items(),
                key=lambda kv: kv[1].get("win_rate_when_selected", 0.0),
                reverse=True,
            )
            for name, _ in ranked:
                try:
                    return Strategy(name)
                except ValueError:
                    continue

        # 3. Conservative default.
        return Strategy.ANALYTICAL

    def _alternative_strategy(self, current: Strategy) -> Strategy:
        """Return a deterministic alternative to the current strategy.

        Used for replanning; picks the first option from a stable preference
        order that differs from `current`.
        """
        order = [
            Strategy.ANALYTICAL,
            Strategy.RETRIEVAL_FIRST,
            Strategy.TOOL_FIRST,
            Strategy.HYBRID,
            Strategy.CREATIVE,
        ]
        for s in order:
            if s != current:
                return s
        return current

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------
    def _default_decomposition(
        self,
        task: Task,
        strategy: Strategy,
        budget: Budget,
        gates: Gates,
    ) -> list[PlanStep]:
        """Produce a simple linear chain of canonical phases.

        Phases are included based on the gating decisions from the Scheduler
        and the reasoning strategy:

            retrieve_context       (if retrieval gate open OR retrieval_first)
            invoke_primary_tool    (if tool_first)
            reason_about_task      (always)
            produce_final_answer   (always)
        """
        phases: list[tuple[str, Optional[str]]] = []

        if gates.retrieval or strategy == Strategy.RETRIEVAL_FIRST:
            phases.append(("retrieve_context", "retrieval"))

        if strategy == Strategy.TOOL_FIRST:
            phases.append(("invoke_primary_tool", "primary_tool"))

        phases.append(("reason_about_task", None))
        phases.append(("produce_final_answer", None))

        per_step = max(256, budget.execution_tokens // max(1, len(phases)))

        steps: list[PlanStep] = []
        prev_id: Optional[str] = None
        for action, tool in phases:
            sid = str(uuid.uuid4())
            steps.append(
                PlanStep(
                    step_id=sid,
                    action=action,
                    strategy=strategy,
                    prompt=self._prompt_for(action, task),
                    tool=tool,
                    depends_on=[prev_id] if prev_id else [],
                    max_tokens=per_step,
                )
            )
            prev_id = sid
        return steps

    def _instantiate_template(
        self,
        template: dict,
        task: Task,
        budget: Budget,
    ) -> list[PlanStep]:
        """Turn a Procedural Memory template into a concrete linear Plan."""
        tpl_steps = template.get("steps", [])
        if not tpl_steps:
            return []
        per_step = max(256, budget.execution_tokens // len(tpl_steps))

        steps: list[PlanStep] = []
        prev_id: Optional[str] = None
        for ts in tpl_steps:
            sid = str(uuid.uuid4())
            strategy_name = ts.get("strategy", "analytical")
            try:
                strategy = Strategy(strategy_name)
            except ValueError:
                strategy = Strategy.ANALYTICAL
            steps.append(
                PlanStep(
                    step_id=sid,
                    action=ts["action"],
                    strategy=strategy,
                    prompt=self._prompt_for(ts["action"], task),
                    tool=ts.get("tool"),
                    depends_on=[prev_id] if prev_id else [],
                    max_tokens=per_step,
                )
            )
            prev_id = sid
        return steps

    def _prompt_for(self, action: str, task: Task) -> str:
        return f"[{action}] Task: {task.input}"
