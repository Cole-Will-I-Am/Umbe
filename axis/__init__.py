"""AXIS v0.2 — Autonomous eXtrospective Intelligence System.

This package implements Priority 1 from the v0.2 engineering spec:
the core inference loop of Executor + Planner + basic Scheduler.

Later priorities (Verifier, Telemetry Observer, Memory Manager, Policy
Router, Self-Model, Adaptation Manager, Eval Harness) are not included
in this module yet.
"""

from .types import (
    Budget,
    ExecutionTrace,
    Gates,
    Plan,
    PlanStep,
    Stakes,
    StepResult,
    Strategy,
    Task,
    TaskOutcome,
)
from .scheduler import Scheduler, SchedulerConfig
from .planner import Planner
from .executor import Executor, WorkingMemory
from .telemetry import StructuredSink, TelemetryObserver, TelemetryRecord
from .runtime import AxisRuntime

__all__ = [
    "Budget",
    "ExecutionTrace",
    "Gates",
    "Plan",
    "PlanStep",
    "Stakes",
    "StepResult",
    "Strategy",
    "Task",
    "TaskOutcome",
    "Scheduler",
    "SchedulerConfig",
    "Planner",
    "Executor",
    "WorkingMemory",
    "TelemetryObserver",
    "TelemetryRecord",
    "StructuredSink",
    "AxisRuntime",
]
