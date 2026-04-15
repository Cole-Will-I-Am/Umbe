"""The Scheduler: AXIS's resource allocator and gating authority.

Spec §1.5, §5, §7. Key invariants enforced here:

* The Scheduler is the only component that can gate retrieval, verification,
  or meta review. If a gate is closed, downstream components must not act.
* The Scheduler cannot set its own budget. Its configuration is supplied from
  outside at construction time (Bounded Autonomy, §5).
* Every allocation respects the hard token ceiling from configuration.

Priority 1 provides a rule-based implementation. A learned policy optimizing
expected_quality / compute_cost is deferred to later priorities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import (
    BUDGET_SPLIT,
    DEFAULT_TASK_BUDGETS,
    HARD_TOKEN_CEILING,
    STAKES_MULTIPLIER,
)
from .types import Budget, Gates, Stakes, Task


@dataclass
class SchedulerConfig:
    task_budgets: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_TASK_BUDGETS))
    stakes_multiplier: dict[str, float] = field(
        default_factory=lambda: dict(STAKES_MULTIPLIER)
    )
    hard_ceiling: int = HARD_TOKEN_CEILING
    split: dict[str, float] = field(default_factory=lambda: dict(BUDGET_SPLIT))
    # Novelty threshold above which retrieval is gated open.
    novelty_retrieval_threshold: float = 0.3
    # Domain confidence below which retrieval is gated open even if novelty is low.
    domain_confidence_retrieval_threshold: float = 0.7
    # Domain error rate above which verification is gated open.
    domain_error_verification_threshold: float = 0.15
    # Strategy win rate below which exploration budget is allocated.
    explore_win_rate_threshold: float = 0.7
    explore_fraction: float = 0.2


class Scheduler:
    """Allocates compute across components and decides gating.

    See spec §7 for the decision procedure. The Scheduler does not execute
    anything; it returns a Budget and a Gates object that the runtime passes
    to the Planner and Executor.
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        self._config = config or SchedulerConfig()

    @property
    def config(self) -> SchedulerConfig:
        """Read-only config view. The Scheduler must not mutate its own budget (§5)."""
        return self._config

    # ------------------------------------------------------------------
    # Budget allocation
    # ------------------------------------------------------------------
    def allocate(self, task: Task, self_model: Optional[dict] = None) -> Budget:
        """Produce a per-task Budget with the 15/55/15/15 split."""
        base = self._config.task_budgets.get(
            task.task_class, self._config.task_budgets.get("general", 4000)
        )

        # Complexity scales the base total in [0.6, 1.6]x.
        complexity = max(0.0, min(task.complexity, 1.0))
        complexity_factor = 0.6 + complexity
        scaled = base * complexity_factor

        multiplier = self._config.stakes_multiplier.get(task.stakes.value, 1.0)
        total = int(scaled * multiplier)
        total = min(total, self._config.hard_ceiling)

        split = self._config.split
        planning = int(total * split["planning"])
        execution = int(total * split["execution"])
        verification = int(total * split["verification"])
        # Absorb rounding drift into the buffer so sub-budgets sum exactly.
        buffer = total - planning - execution - verification

        return Budget(
            total_tokens=total,
            planning_tokens=planning,
            execution_tokens=execution,
            verification_tokens=verification,
            buffer_tokens=buffer,
            deadline_ms=task.deadline_ms,
        )

    # ------------------------------------------------------------------
    # Gating decisions
    # ------------------------------------------------------------------
    def gates(
        self,
        task: Task,
        self_model: Optional[dict] = None,
        novelty: float = 0.5,
    ) -> Gates:
        """Decide retrieval / verification / exploration gates for this task."""
        self_model = self_model or {}
        cap = self_model.get("capabilities", {}).get(task.task_class, {})
        domain_confidence = float(cap.get("win_rate", 0.5))
        domain_error = 1.0 - domain_confidence

        retrieval_open = (
            novelty > self._config.novelty_retrieval_threshold
            or domain_confidence < self._config.domain_confidence_retrieval_threshold
        )

        verification_open = (
            task.stakes in (Stakes.HIGH, Stakes.CRITICAL)
            or domain_error > self._config.domain_error_verification_threshold
        )
        # Critical always verifies, regardless of anything else.
        if task.stakes == Stakes.CRITICAL:
            verification_open = True
        # Low-stakes tasks skip verification to save compute unless error rate
        # is clearly bad for this domain.
        if task.stakes == Stakes.LOW and domain_error <= self._config.domain_error_verification_threshold:
            verification_open = False

        exploration = 0.0
        if (
            domain_confidence < self._config.explore_win_rate_threshold
            and task.stakes in (Stakes.LOW, Stakes.MEDIUM)
        ):
            exploration = self._config.explore_fraction

        return Gates(
            retrieval=retrieval_open,
            verification=verification_open,
            meta_trigger=False,  # Meta trigger wiring arrives with Telemetry Observer.
            exploration_fraction=exploration,
        )

    # ------------------------------------------------------------------
    # Stop condition (marginal value of computation, §7.2)
    # ------------------------------------------------------------------
    def should_continue(
        self,
        tokens_used: int,
        budget: Budget,
        marginal_quality_gain: float,
        marginal_cost_estimate: float,
        latency_ms: int,
    ) -> bool:
        """Return True iff further computation is worth its cost."""
        if tokens_used >= budget.total_tokens:
            return False
        if budget.deadline_ms is not None and latency_ms >= budget.deadline_ms:
            return False
        if marginal_quality_gain <= marginal_cost_estimate:
            return False
        return True
