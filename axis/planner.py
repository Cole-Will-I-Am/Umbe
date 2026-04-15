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

import json
import re
import uuid
from typing import Callable, Optional

from .backends.base import InferenceBackend
from .config import DEFAULT_SELF_MODEL
from .types import Budget, Gates, Plan, PlanStep, Strategy, Task
from .world_model import TypedState, WorldModel

# Retriever contract: (task_input, task_class) -> retrieved-context string.
# Return an empty string when there is nothing useful to add — the
# Planner inlines whatever it gets.
Retriever = Callable[[str, str], str]


class Planner:
    """Builds a DAG of PlanSteps and selects the reasoning strategy."""

    def __init__(
        self,
        self_model: Optional[dict] = None,
        procedural_memory: Optional[dict] = None,
        backend: Optional[InferenceBackend] = None,
        world_model: Optional[WorldModel] = None,
        llm_candidates_k: int = 3,
        llm_max_tokens: int = 1024,
        retriever: Optional[Retriever] = None,
    ):
        # Deep-copy-lite: shallow copies are fine here since Priority 1
        # treats both as read-only.
        self.self_model = self_model if self_model is not None else dict(DEFAULT_SELF_MODEL)
        self.procedural_memory = procedural_memory or {}
        # Optional LLM-driven decomposition path. When ``backend`` is
        # None the Planner stays in its rule-based, zero-dependency
        # mode — that's the path every default-constructed Planner()
        # in the existing test suite exercises.
        self.backend = backend
        self.world_model = world_model
        self.llm_candidates_k = llm_candidates_k
        self.llm_max_tokens = llm_max_tokens
        # Optional retriever: given task_input + task_class, returns a
        # short string of retrieved context. The Planner injects the
        # result into the prompt of any ``retrieve_context`` step it
        # emits. Default None → no-op, behaviour unchanged.
        self.retriever = retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan(self, task: Task, budget: Budget, gates: Gates) -> Plan:
        """Produce an initial Plan for a task.

        Precedence:
          1. Procedural Memory template for the task class, if present.
          2. LLM-driven decomposition (if a backend is configured),
             optionally filtered through the WorldModel's feasibility
             check.
          3. Rule-based default decomposition (the Priority 1 fallback).
        """
        strategy = self._choose_strategy(task)

        template = self.procedural_memory.get(task.task_class)
        if template:
            steps = self._instantiate_template(template, task, budget)
            rationale = f"procedural_template:{template.get('name', task.task_class)}"
            return Plan(
                task_id=task.task_id,
                steps=steps,
                strategy=strategy,
                budget=budget,
                gates=gates,
                rationale=rationale,
            )

        if self.backend is not None:
            llm_plan = self._llm_decomposition(task, strategy, budget, gates)
            if llm_plan is not None:
                return llm_plan

        steps = self._default_decomposition(task, strategy, budget, gates)
        return Plan(
            task_id=task.task_id,
            steps=steps,
            strategy=strategy,
            budget=budget,
            gates=gates,
            rationale=f"default_decomposition:{strategy.value}",
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
        base = f"[{action}] Task: {task.input}"
        if action == "retrieve_context" and self.retriever is not None:
            try:
                retrieved = self.retriever(task.input, task.task_class) or ""
            except Exception:
                retrieved = ""
            if retrieved:
                base = (
                    f"[{action}] Task: {task.input}\n"
                    f"[retrieved_context]\n{retrieved}\n[/retrieved_context]"
                )
        return base

    # ------------------------------------------------------------------
    # LLM-driven decomposition (§1.2)
    # ------------------------------------------------------------------
    LLM_PROMPT_TEMPLATE = (
        "You are AXIS's Planner. Break the following task into an "
        "executable DAG of 2–6 steps. Respond with ONLY a single JSON "
        "object (no prose, no markdown) of the form:\n"
        '{{"steps": [{{"id": "s1", "action": "...", "tool": null, '
        '"strategy": "analytical", "depends_on": []}}, ...]}}\n\n'
        "Rules:\n"
        "- Each step id must be unique.\n"
        "- depends_on lists upstream step ids (may be empty).\n"
        "- strategy must be one of: analytical, creative, "
        "retrieval_first, tool_first, hybrid.\n"
        "- tool is either null or a short tool name.\n"
        "- The DAG must be acyclic and have exactly one terminal step.\n\n"
        "Task class: {task_class}\n"
        "Preferred strategy: {strategy}\n"
        "Task: {task_input}\n"
    )

    def _llm_decomposition(
        self,
        task: Task,
        strategy: Strategy,
        budget: Budget,
        gates: Gates,
    ) -> Optional[Plan]:
        """Ask the backend for a JSON DAG; parse into PlanSteps.

        Up to ``llm_candidates_k`` samples are drawn. If a WorldModel is
        wired in, infeasible candidates are filtered out and the
        cheapest feasible one (by summed estimated_latency_ms) wins.
        Returns None on total failure so the caller can fall back to
        the rule-based decomposition.
        """
        assert self.backend is not None
        prompt = self.LLM_PROMPT_TEMPLATE.format(
            task_class=task.task_class,
            strategy=strategy.value,
            task_input=task.input,
        )

        candidates: list[list[PlanStep]] = []
        max_k = max(1, self.llm_candidates_k)
        for _ in range(max_k):
            try:
                result = self.backend.run(
                    prompt=prompt,
                    max_tokens=self.llm_max_tokens,
                )
            except Exception:
                continue
            if not result.success or not result.output:
                continue
            steps = self._parse_llm_plan(result.output, task, strategy, budget)
            if steps:
                candidates.append(steps)

        if not candidates:
            return None

        chosen = self._pick_best_candidate(candidates, task)
        if chosen is None:
            return None

        return Plan(
            task_id=task.task_id,
            steps=chosen,
            strategy=strategy,
            budget=budget,
            gates=gates,
            rationale=f"llm_decomposition:{strategy.value}",
        )

    def _parse_llm_plan(
        self,
        raw: str,
        task: Task,
        strategy: Strategy,
        budget: Budget,
    ) -> Optional[list[PlanStep]]:
        """Parse a JSON-shaped LLM plan. Tolerant of code fences and
        surrounding prose; extracts the first balanced JSON object."""
        blob = _extract_json_object(raw)
        if blob is None:
            return None
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            return None
        raw_steps = data.get("steps") if isinstance(data, dict) else None
        if not isinstance(raw_steps, list) or not raw_steps:
            return None

        # Pass 1: allocate a stable UUID per declared id so
        # depends_on references resolve.
        id_map: dict[str, str] = {}
        for ts in raw_steps:
            if not isinstance(ts, dict):
                return None
            raw_id = str(ts.get("id") or uuid.uuid4())
            id_map[raw_id] = str(uuid.uuid4())

        per_step = max(256, budget.execution_tokens // len(raw_steps))

        # Pass 2: build PlanSteps with resolved dependencies.
        steps: list[PlanStep] = []
        for ts in raw_steps:
            raw_id = str(ts.get("id") or "")
            sid = id_map.get(raw_id) or str(uuid.uuid4())
            action = str(ts.get("action") or "step").strip() or "step"
            tool = ts.get("tool")
            if tool is not None:
                tool = str(tool)
            try:
                step_strategy = Strategy(str(ts.get("strategy") or strategy.value))
            except ValueError:
                step_strategy = strategy
            depends_raw = ts.get("depends_on") or []
            if not isinstance(depends_raw, list):
                depends_raw = []
            depends_on = [
                id_map[str(d)] for d in depends_raw if str(d) in id_map
            ]
            steps.append(
                PlanStep(
                    step_id=sid,
                    action=action,
                    strategy=step_strategy,
                    prompt=self._prompt_for(action, task),
                    tool=tool,
                    depends_on=depends_on,
                    max_tokens=per_step,
                )
            )

        # Validate: no cycles, connected DAG (topological_order raises
        # on cycles / unknown deps).
        from .types import Plan as _Plan  # local import to avoid cycle

        probe = _Plan(
            task_id=task.task_id,
            steps=steps,
            strategy=strategy,
            budget=budget,
            gates=Gates(),
        )
        try:
            probe.topological_order()
        except ValueError:
            return None
        return steps

    def _pick_best_candidate(
        self,
        candidates: list[list[PlanStep]],
        task: Task,
    ) -> Optional[list[PlanStep]]:
        """If a WorldModel is configured, keep only feasible candidates
        and rank by estimated latency. Otherwise return the first."""
        if self.world_model is None:
            return candidates[0]

        initial = task.metadata.get("initial_state") or {}
        start = TypedState(values=dict(initial))
        feasible_ranked: list[tuple[int, list[PlanStep]]] = []
        for cand in candidates:
            action_sequence = [s.action for s in cand]
            transitions = self.world_model.simulate_sequence(start, action_sequence)
            feasible = (
                len(transitions) == len(action_sequence)
                and all(t.succeeded for t in transitions)
            )
            if not feasible:
                continue
            latency = 0
            for action in action_sequence:
                schema = self.world_model.get(action)
                if schema is not None:
                    latency += schema.estimated_latency_ms
            feasible_ranked.append((latency, cand))

        if not feasible_ranked:
            # No feasible candidate — caller will fall back.
            return None
        feasible_ranked.sort(key=lambda t: t[0])
        return feasible_ranked[0][1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_json_object(raw: str) -> Optional[str]:
    """Return the first balanced top-level JSON object substring in raw.

    Handles code-fenced blocks, leading/trailing prose, and escaped
    quotes inside strings. Returns None if no object is found.
    """
    # Strip fenced code blocks if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = raw.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None
