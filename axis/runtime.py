"""AxisRuntime: the Priority-1 integration of Scheduler, Planner, and Executor.

This wires the three core components from spec §13 Priority 1. Later
priorities will insert additional stages into this loop:

    Priority 2 — Telemetry Observer after every execution
    Priority 3 — Working / Episodic Memory persistence
    Priority 4 — Verifier between execution and commit
    Priority 5 — Policy Router before planning
    Priority 8 — Adaptation Manager feedback into Planner / Scheduler

The runtime is intentionally small and blocking; concurrency arrives with
the Multi-Instance Coordination layer (Priority 10).
"""

from __future__ import annotations

from typing import Optional

from .executor import Executor, WorkingMemory
from .memory import MemoryManager
from .planner import Planner
from .policy_router import PolicyRouter, RoutingDecision
from .scheduler import Scheduler
from .telemetry import TelemetryObserver
from .types import ExecutionTrace, Task, TaskOutcome
from .verifier import VerificationResult, Verifier


class AxisRuntime:
    """Top-level driver for the core inference loop."""

    def __init__(
        self,
        scheduler: Optional[Scheduler] = None,
        planner: Optional[Planner] = None,
        executor: Optional[Executor] = None,
        observer: Optional[TelemetryObserver] = None,
        memory: Optional[MemoryManager] = None,
        verifier: Optional[Verifier] = None,
        policy_router: Optional[PolicyRouter] = None,
        max_replans: int = 2,
        auto_refresh_self_model: bool = True,
    ) -> None:
        self.scheduler = scheduler or Scheduler()
        self.planner = planner or Planner()
        self.executor = executor or Executor()
        self.observer = observer
        self.memory = memory
        self.verifier = verifier
        self.policy_router = policy_router
        self.max_replans = max_replans
        self.auto_refresh_self_model = auto_refresh_self_model

    def run(
        self,
        task: Task,
        tool_inventory: Optional[dict] = None,
    ) -> ExecutionTrace:
        """Run a task end-to-end and return its ExecutionTrace.

        On failure, the Planner is asked to replan up to `max_replans`
        times. The Executor never improvises — all recovery goes through
        the Planner (spec §1.2). After the run completes, the trace is
        handed to the TelemetryObserver (if configured); the Observer
        is the only component permitted to write the Self-Model (§6).
        """
        # Policy Router (optional) decides strategy / verification depth /
        # retrieval posture before the Scheduler allocates budget.
        routing_decision: Optional[RoutingDecision] = None
        if self.policy_router is not None:
            routing_decision = self.policy_router.route(
                task, self_model=self.planner.self_model
            )

        # Scheduler next: it owns budget and gating.
        budget = self.scheduler.allocate(task, self.planner.self_model)
        gates = self.scheduler.gates(task, self.planner.self_model)
        # Router can force the verification gate open/closed — but never
        # below the safety floor (Verifier.verify still enforces §5
        # Honesty on CRITICAL stakes).
        if routing_decision is not None:
            if routing_decision.verification_depth == "skip":
                gates.verification = False
            elif routing_decision.verification_depth == "full":
                gates.verification = True

        # Planner turns the task into an executable DAG.
        plan = self.planner.plan(task, budget, gates)
        # If the Policy Router chose a strategy, override the Planner's
        # fallback choice on the plan object so downstream telemetry
        # reflects the routing decision.
        if routing_decision is not None:
            plan.strategy = routing_decision.strategy

        # Executor carries it out in its own fresh Working Memory.
        wm = WorkingMemory()
        trace = self.executor.execute_plan(plan, wm, tool_inventory)

        replans = 0
        while (
            trace.outcome == TaskOutcome.FAILURE
            and replans < self.max_replans
            and trace.step_results
        ):
            failed = trace.step_results[-1]
            plan = self.planner.replan(
                task=task,
                plan=plan,
                failed_step_id=failed.step_id,
                error=failed.error or "unknown",
            )
            replans += 1

            # Fresh Working Memory per replan attempt — per spec §1.6.1 the
            # scratchpad is task-scoped and gets destroyed on retry.
            wm = WorkingMemory()
            trace = self.executor.execute_plan(plan, wm, tool_inventory)
            trace.replanning_events = replans

        # Fill in fields the trace can only know with the task in hand so
        # the observer gets a faithful record.
        trace.task_class = task.task_class
        if trace.strategy is None:
            trace.strategy = plan.strategy
        if routing_decision is not None:
            trace.routing_decision = routing_decision.to_dict()

        # Verifier: scores the execution, never mutates it. Control
        # returns to the Planner on failing score (replan once more).
        if self.verifier is not None:
            result = self.verifier.verify(task, trace, gate_open=gates.verification)
            trace.verifier_score = result.score
            trace.confidence_predicted = result.confidence
            trace.verification_tokens = 0  # stub backend, no token charge
            if not result.passed and not result.skipped:
                # Single verifier-driven replan attempt — bounded so we
                # don't loop forever on a pathologically failing task.
                if trace.replanning_events < self.max_replans and trace.step_results:
                    failed = trace.step_results[-1]
                    plan = self.planner.replan(
                        task=task,
                        plan=plan,
                        failed_step_id=failed.step_id,
                        error=f"verifier_rejected:{result.rationale}",
                    )
                    trace.replanning_events += 1
                    wm = WorkingMemory()
                    new_trace = self.executor.execute_plan(plan, wm, tool_inventory)
                    new_trace.task_class = task.task_class
                    new_trace.strategy = plan.strategy
                    new_trace.replanning_events = trace.replanning_events
                    if routing_decision is not None:
                        new_trace.routing_decision = routing_decision.to_dict()
                    # Re-verify the new attempt
                    result2 = self.verifier.verify(
                        task, new_trace, gate_open=gates.verification
                    )
                    new_trace.verifier_score = result2.score
                    new_trace.confidence_predicted = result2.confidence
                    trace = new_trace

        if self.policy_router is not None and routing_decision is not None:
            self.policy_router.record_outcome(task.task_id, trace.outcome.value)

        if self.observer is not None:
            self.observer.record_trace(trace, task, plan)
            if (
                self.auto_refresh_self_model
                and self.observer.should_refresh_self_model()
            ):
                self._refresh_self_model()

        if self.memory is not None:
            self.memory.record_episode(trace, task)
            # Keep the Planner's procedural-memory view in sync with the
            # MemoryManager's validated procedures.
            self.planner.procedural_memory = self.memory.procedural_index_for_planner()

        return trace

    def _refresh_self_model(self) -> None:
        """Pull a fresh Self-Model snapshot from the Observer into the Planner.

        Per spec §6, only the Telemetry Observer may write to the Self-Model.
        This method is the one-way pipe from Observer → Planner; no other
        path exists.
        """
        assert self.observer is not None
        snapshot = self.observer.self_model_snapshot()
        # Preserve any fields the snapshot doesn't yet populate (e.g. the
        # bootstrap architecture dict) by overlaying rather than replacing.
        merged = dict(self.planner.self_model)
        merged.update(snapshot)
        self.planner.self_model = merged
