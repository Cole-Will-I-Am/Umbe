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
from .telemetry import TelemetryObserver
from .types import ExecutionTrace, Task, TaskOutcome


class AxisRuntime:
    """Top-level driver for the core inference loop."""

    def __init__(
        self,
        scheduler: Optional[Scheduler] = None,
        planner: Optional[Planner] = None,
        executor: Optional[Executor] = None,
        observer: Optional[TelemetryObserver] = None,
        max_replans: int = 2,
        auto_refresh_self_model: bool = True,
    ) -> None:
        self.scheduler = scheduler or Scheduler()
        self.planner = planner or Planner()
        self.executor = executor or Executor()
        self.observer = observer
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

        # Fill in fields the trace can only know with the task in hand so
        # the observer gets a faithful record.
        trace.task_class = task.task_class
        if trace.strategy is None:
            trace.strategy = plan.strategy

        if self.observer is not None:
            self.observer.record_trace(trace, task, plan)
            if (
                self.auto_refresh_self_model
                and self.observer.should_refresh_self_model()
            ):
                self._refresh_self_model()

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
