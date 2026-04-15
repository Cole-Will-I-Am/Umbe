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
from .memory import (
    Episode,
    EpisodicMemory,
    ForgettingEngine,
    MemoryManager,
    Procedure,
    ProceduralMemory,
    SemanticEntity,
    SemanticMemory,
)
from .adaptation import (
    AdaptationConfig,
    AdaptationManager,
    ChangeClass,
    Deployment,
    Proposal,
)
from .eval_harness import EvalCase, EvalHarness, EvalReport
from .policy_router import PolicyRouter, RoutingDecision
from .safety import InvariantViolation, SafetyMonitor
from .telemetry import StructuredSink, TelemetryObserver, TelemetryRecord
from .verifier import VerificationResult, Verifier, VerifierConfig
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
    "MemoryManager",
    "EpisodicMemory",
    "Episode",
    "SemanticMemory",
    "SemanticEntity",
    "ProceduralMemory",
    "Procedure",
    "ForgettingEngine",
    "Verifier",
    "VerifierConfig",
    "VerificationResult",
    "PolicyRouter",
    "RoutingDecision",
    "AdaptationManager",
    "AdaptationConfig",
    "ChangeClass",
    "Deployment",
    "Proposal",
    "EvalCase",
    "EvalHarness",
    "EvalReport",
    "SafetyMonitor",
    "InvariantViolation",
    "AxisRuntime",
]
