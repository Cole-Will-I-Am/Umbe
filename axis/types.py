"""Core data types for the AXIS runtime.

These are the shared structures that flow between Scheduler, Planner,
Executor, and (later) Verifier / Telemetry. Keep them dependency-free.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Strategy(str, Enum):
    """Reasoning strategies the Policy Router / Planner can select.

    Matches the routing options from spec §1.4.
    """

    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    RETRIEVAL_FIRST = "retrieval_first"
    TOOL_FIRST = "tool_first"
    HYBRID = "hybrid"


class Stakes(str, Enum):
    """Stakes levels used by the Scheduler for dynamic objective adjustment (§3.3)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskOutcome(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


@dataclass
class Task:
    """A unit of work submitted to the runtime."""

    input: str
    task_class: str = "general"
    stakes: Stakes = Stakes.MEDIUM
    complexity: float = 0.5  # 0.0 trivial .. 1.0 hardest
    deadline_ms: Optional[int] = None
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Budget:
    """Compute budget produced by the Scheduler for a single task.

    Follows the 15/55/15/15 split from spec §7.1.
    """

    total_tokens: int
    planning_tokens: int
    execution_tokens: int
    verification_tokens: int
    buffer_tokens: int
    deadline_ms: Optional[int] = None

    def sub_budget_for(self, phase: str) -> int:
        return {
            "planning": self.planning_tokens,
            "execution": self.execution_tokens,
            "verification": self.verification_tokens,
            "buffer": self.buffer_tokens,
        }[phase]


@dataclass
class Gates:
    """Gating decisions set by the Scheduler (§1.5, §7.1).

    Components may only perform a gated action when the gate is open.
    """

    retrieval: bool = False
    verification: bool = False
    meta_trigger: bool = False
    exploration_fraction: float = 0.0


@dataclass
class PlanStep:
    """A single node in a Plan DAG. Produced by Planner, consumed by Executor."""

    step_id: str
    action: str
    strategy: Strategy
    prompt: str
    tool: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    max_tokens: int = 512
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """A directed acyclic graph of PlanSteps plus the budget/gates it runs under."""

    task_id: str
    steps: list[PlanStep]
    strategy: Strategy
    budget: Budget
    gates: Gates
    rationale: str = ""

    def topological_order(self) -> list[PlanStep]:
        """Return steps in a dependency-respecting order. Raises on cycle."""
        id_to_step = {s.step_id: s for s in self.steps}
        indeg: dict[str, int] = {s.step_id: 0 for s in self.steps}
        for s in self.steps:
            for dep in s.depends_on:
                if dep not in id_to_step:
                    raise ValueError(f"step {s.step_id} depends on unknown {dep}")
                indeg[s.step_id] += 1

        queue = [sid for sid, d in indeg.items() if d == 0]
        order: list[PlanStep] = []
        while queue:
            sid = queue.pop(0)
            order.append(id_to_step[sid])
            for s in self.steps:
                if sid in s.depends_on:
                    indeg[s.step_id] -= 1
                    if indeg[s.step_id] == 0:
                        queue.append(s.step_id)

        if len(order) != len(self.steps):
            raise ValueError("plan contains a cycle")
        return order


@dataclass
class StepResult:
    """Result of executing one PlanStep."""

    step_id: str
    output: str
    tokens_used: int
    tool_calls: list[dict] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    entropy: float = 0.0


@dataclass
class ExecutionTrace:
    """End-to-end record of running a Plan. Consumed by the Telemetry Observer.

    Fields populated by Priority-1 components:
        step_results, total_tokens, replanning_events, outcome, error,
        timing, strategy (copied from the final Plan).

    Fields reserved for later priorities (remain at their defaults until
    those components ship):
        verifier_score        — Priority 4 Verifier
        confidence_predicted  — Priority 4 Verifier
        routing_decision      — Priority 5 Policy Router
        user_correction       — Priority 3 post-task feedback channel
    """

    task_id: str
    task_class: str = "general"
    strategy: Optional[Strategy] = None
    step_results: list[StepResult] = field(default_factory=list)
    planning_tokens: int = 0
    execution_tokens: int = 0
    verification_tokens: int = 0
    total_tokens: int = 0
    replanning_events: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    outcome: TaskOutcome = TaskOutcome.SUCCESS
    error: Optional[str] = None
    verifier_score: Optional[float] = None
    confidence_predicted: Optional[float] = None
    routing_decision: Optional[dict] = None
    user_correction: Optional[bool] = None

    def latency_ms(self) -> int:
        end = self.end_time if self.end_time is not None else time.time()
        return int((end - self.start_time) * 1000)

    def tool_call_summary(self) -> list[dict]:
        """Flatten tool calls across all steps for telemetry emission."""
        out: list[dict] = []
        for sr in self.step_results:
            for tc in sr.tool_calls:
                out.append(
                    {
                        "tool": tc.get("tool"),
                        "success": tc.get("success", True),
                        "error": tc.get("error"),
                        "latency_ms": tc.get("latency_ms", 0),
                    }
                )
        return out
