"""Policy Router — spec §1.4.

Given a task + Objective Stack + Self-Model, decides:

    reasoning_strategy       analytical | creative | retrieval_first |
                             tool_first | hybrid
    verification_depth       skip | spot_check | full
    retrieval_policy         none | targeted | broad_sweep
    response_posture         concise | thorough | exploratory | cautious
    escalation_threshold     verifier score below which to re-plan

Every routing decision is logged with its outcome so the dataset for
learned routing (Priority 8+) accumulates from day one. The Router
starts as a prompted/rule-based decision with structured output and
graduates to a trained model as data accumulates.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from .types import Stakes, Strategy, Task


@dataclass
class RoutingDecision:
    strategy: Strategy
    verification_depth: str  # skip | spot_check | full
    retrieval_policy: str  # none | targeted | broad_sweep
    response_posture: str  # concise | thorough | exploratory | cautious
    escalation_threshold: float
    rationale: str
    decided_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "strategy": self.strategy.value,
        }


class PolicyRouter:
    """Rule-based v1. The interface is stable enough that the learned v2
    (trained on accumulated routing logs) can slot in without changes to
    callers."""

    def __init__(self) -> None:
        self._log: list[tuple[dict, RoutingDecision, Optional[str]]] = []

    def route(
        self,
        task: Task,
        self_model: Optional[dict] = None,
        objective_stack: Optional[dict] = None,
    ) -> RoutingDecision:
        self_model = self_model or {}
        objective_stack = objective_stack or {}

        cap = (self_model.get("capabilities") or {}).get(task.task_class) or {}
        domain_win_rate = float(cap.get("win_rate", 0.5))
        best_strategy_name = cap.get("best_strategy")

        # Strategy selection
        if best_strategy_name:
            try:
                strategy = Strategy(best_strategy_name)
            except ValueError:
                strategy = self._fallback_strategy(self_model)
        else:
            strategy = self._fallback_strategy(self_model)

        # Verification depth scales with stakes and domain error rate
        if task.stakes == Stakes.CRITICAL:
            verification_depth = "full"
        elif task.stakes == Stakes.HIGH or domain_win_rate < 0.6:
            verification_depth = "full"
        elif task.stakes == Stakes.LOW and domain_win_rate > 0.85:
            verification_depth = "skip"
        else:
            verification_depth = "spot_check"

        # Retrieval policy
        if strategy == Strategy.RETRIEVAL_FIRST:
            retrieval_policy = "broad_sweep"
        elif domain_win_rate < 0.7:
            retrieval_policy = "targeted"
        else:
            retrieval_policy = "none"

        # Response posture
        if task.stakes in (Stakes.HIGH, Stakes.CRITICAL):
            posture = "cautious"
        elif task.stakes == Stakes.LOW:
            posture = "concise"
        elif domain_win_rate < 0.7:
            posture = "exploratory"
        else:
            posture = "thorough"

        # Escalation threshold — higher stakes demand stricter Verifier passing
        stakes_to_threshold = {
            Stakes.LOW: 0.45,
            Stakes.MEDIUM: 0.60,
            Stakes.HIGH: 0.75,
            Stakes.CRITICAL: 0.85,
        }
        escalation_threshold = stakes_to_threshold[task.stakes]

        decision = RoutingDecision(
            strategy=strategy,
            verification_depth=verification_depth,
            retrieval_policy=retrieval_policy,
            response_posture=posture,
            escalation_threshold=escalation_threshold,
            rationale=(
                f"stakes={task.stakes.value} domain_win={domain_win_rate:.2f} "
                f"best_strategy={best_strategy_name or 'fallback'}"
            ),
        )
        features = {
            "task_class": task.task_class,
            "stakes": task.stakes.value,
            "complexity": task.complexity,
            "domain_win_rate": domain_win_rate,
            "best_strategy": best_strategy_name,
        }
        self._log.append((features, decision, None))
        return decision

    def record_outcome(self, task_id: str, outcome: str) -> None:
        """Associate the most recent pending routing decision with an outcome.

        Called by the runtime after a task completes. The features +
        decision + outcome tuple is the training data for the learned
        router upgrade.
        """
        for i in range(len(self._log) - 1, -1, -1):
            features, decision, existing = self._log[i]
            if existing is None:
                self._log[i] = (features, decision, outcome)
                return

    def training_data(self) -> list[dict]:
        """Return (features, decision, outcome) tuples as a dataset."""
        out: list[dict] = []
        for features, decision, outcome in self._log:
            if outcome is None:
                continue
            out.append(
                {
                    "features": features,
                    "decision": decision.to_dict(),
                    "outcome": outcome,
                }
            )
        return out

    def _fallback_strategy(self, self_model: dict) -> Strategy:
        eff = self_model.get("strategy_effectiveness") or {}
        if not eff:
            return Strategy.ANALYTICAL
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
        return Strategy.ANALYTICAL
