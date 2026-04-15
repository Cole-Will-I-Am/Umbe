"""The Executor: runs inference on fully specified PlanSteps.

Spec §1.1. Invariants:

* The Executor does not choose strategy, decide whether to verify, manage
  memory beyond Working Memory, or evaluate its own performance.
* A hard token ceiling per execution call is enforced regardless of what
  the backend reports.
* Failures return control to the Planner for replanning; the Executor
  never improvises.
* Writes only to Working Memory. No Episodic / Semantic / Procedural /
  Self-Model access (§6 State Access Matrix).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from .backends.base import BackendResult, InferenceBackend
from .backends.stub import StubBackend
from .types import ExecutionTrace, Plan, PlanStep, StepResult, TaskOutcome


class WorkingMemory:
    """Per-task scratchpad.

    Destroyed at task completion per spec §1.6.1. For Priority 1 this is
    simply an in-process dict; the Memory Manager (Priority 3 / §1.6) will
    add persistence, spill-to-disk, and summarization.
    """

    def __init__(self) -> None:
        self.scratchpad: dict[str, Any] = {}
        self.step_outputs: dict[str, str] = {}
        self.tool_calls: list[dict] = []

    def put(self, key: str, value: Any) -> None:
        self.scratchpad[key] = value

    def get(self, key: str) -> Any:
        return self.scratchpad.get(key)


class Executor:
    """Runs PlanSteps via an InferenceBackend.

    The Executor is deliberately thin. All decision-making lives in the
    Planner, Scheduler, and (later) Policy Router / Verifier.
    """

    def __init__(self, backend: Optional[InferenceBackend] = None) -> None:
        self.backend: InferenceBackend = backend or StubBackend()

    # ------------------------------------------------------------------
    # Single-step execution
    # ------------------------------------------------------------------
    def execute_step(
        self,
        step: PlanStep,
        working_memory: WorkingMemory,
        tool_inventory: Optional[dict] = None,
    ) -> StepResult:
        tool_inventory = tool_inventory or {}
        context = self._build_context(step, working_memory)

        try:
            result: BackendResult = self.backend.run(
                prompt=context,
                max_tokens=step.max_tokens,
                tool=step.tool,
                tool_inventory=tool_inventory,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return StepResult(
                step_id=step.step_id,
                output="",
                tokens_used=0,
                success=False,
                error=f"backend_exception: {exc!r}",
            )

        # Hard token ceiling (§1.1): the caller promised max_tokens, we hold
        # the backend to it even if it reports more.
        tokens = max(0, min(result.tokens_used, step.max_tokens))

        step_result = StepResult(
            step_id=step.step_id,
            output=result.output,
            tokens_used=tokens,
            tool_calls=list(result.tool_calls or []),
            success=result.success,
            error=result.error,
            entropy=result.entropy,
        )

        if step_result.success:
            working_memory.step_outputs[step.step_id] = step_result.output
            working_memory.tool_calls.extend(step_result.tool_calls)

        return step_result

    # ------------------------------------------------------------------
    # Whole-plan execution
    # ------------------------------------------------------------------
    def execute_plan(
        self,
        plan: Plan,
        working_memory: Optional[WorkingMemory] = None,
        tool_inventory: Optional[dict] = None,
    ) -> ExecutionTrace:
        wm = working_memory or WorkingMemory()
        trace = ExecutionTrace(task_id=plan.task_id)

        try:
            order = plan.topological_order()
        except ValueError as exc:
            trace.outcome = TaskOutcome.FAILURE
            trace.error = f"invalid_plan: {exc}"
            trace.end_time = time.time()
            return trace

        for step in order:
            # Respect the Scheduler's execution sub-budget. We don't look at
            # total_tokens here because the Planner may have allocated the
            # verification/buffer slices for other components.
            if trace.total_tokens >= plan.budget.execution_tokens:
                trace.outcome = TaskOutcome.PARTIAL
                trace.error = "execution_budget_exhausted"
                break

            result = self.execute_step(step, wm, tool_inventory)
            trace.step_results.append(result)
            trace.total_tokens += result.tokens_used

            if not result.success:
                trace.outcome = TaskOutcome.FAILURE
                trace.error = result.error
                break
        else:
            trace.outcome = TaskOutcome.SUCCESS

        trace.end_time = time.time()
        return trace

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_context(self, step: PlanStep, wm: WorkingMemory) -> str:
        """Concatenate upstream step outputs with the step's own prompt.

        Priority 1 uses plain string concatenation. A real backend will
        need a proper context assembly pass respecting the model's
        context window; that's the Planner/Scheduler's job to bound.
        """
        parts: list[str] = []
        for dep_id in step.depends_on:
            dep_output = wm.step_outputs.get(dep_id)
            if dep_output:
                parts.append(f"[dep:{dep_id[:8]}] {dep_output}")
        parts.append(step.prompt)
        return "\n".join(parts)
