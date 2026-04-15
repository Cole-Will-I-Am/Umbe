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
from .planner import Planner
from .scheduler import Scheduler
from .types import ExecutionTrace, Task, TaskOutcome


class AxisRuntime:
    """Top-level driver for the core inference loop."""

    def __init__(
        self,
        scheduler: Optional[Scheduler] = None,
        planner: Optional[Planner] = None,
        executor: Optional[Executor] = None,
        max_replans: int = 2,
    ) -> None:
        self.scheduler = scheduler or Scheduler()
        self.planner = planner or Planner()
        self.executor = executor or Executor()
        self.max_replans = max_replans

    def run(
        self,
        task: Task,
        tool_inventory: Optional[dict] = None,
    ) -> ExecutionTrace:
        """Run a task end-to-end and return its ExecutionTrace.

        On failure, the Planner is asked to replan up to `max_replans`
        times. The Executor never improvises — all recovery goes through
        the Planner (spec §1.2).
        """
        # Scheduler first: it owns budget and gating.
        budget = self.scheduler.allocate(task, self.planner.self_model)
        gates = self.scheduler.gates(task, self.planner.self_model)

        # Planner turns the task into an executable DAG.
        plan = self.planner.plan(task, budget, gates)

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

        return trace
