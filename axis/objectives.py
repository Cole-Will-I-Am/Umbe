"""Objective Stack — spec §3.

Defines what AXIS is optimizing. Supports:

    * Default global weights (§3.1)
    * Per-domain overrides (§3.2)
    * Stakes-based dynamic adjustment (§3.3)
    * Conflict resolution procedure (§3.4)

Hard constraints (safety invariants from §5) are not objectives. They
are absolute boundaries enforced by SafetyMonitor; this module deals
only with weighted trade-offs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .config import DEFAULT_OBJECTIVE_STACK
from .types import Stakes, Task

# Per-domain overrides from spec §3.2. These are partial overlays —
# absent keys fall back to the global defaults.
DEFAULT_DOMAIN_OVERRIDES: dict[str, dict[str, float]] = {
    "medical": {"truthfulness": 0.45, "calibration": 0.25, "originality": 0.00},
    "creative_writing": {
        "originality": 0.20,
        "truthfulness": 0.10,
        "task_success": 0.30,
    },
    "real_time_ops": {
        "latency": 0.25,
        "efficiency": 0.15,
        "truthfulness": 0.20,
    },
    "research": {
        "learning_value": 0.10,
        "robustness": 0.10,
        "truthfulness": 0.25,
    },
}


@dataclass
class ConflictResolution:
    chosen: str
    score: float
    margin: float
    used_tiebreaker: bool = False
    exploration_allocated: bool = False
    runners_up: list[tuple[str, float]] = field(default_factory=list)


class ObjectiveStack:
    """Weighted objective aggregator with stakes + domain adjustments."""

    def __init__(
        self,
        defaults: Optional[dict[str, float]] = None,
        domain_overrides: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        self._defaults = dict(defaults or DEFAULT_OBJECTIVE_STACK)
        self._domain_overrides = dict(domain_overrides or DEFAULT_DOMAIN_OVERRIDES)

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------
    def weights_for(self, task: Task) -> dict[str, float]:
        """Return the effective per-task weights after domain overrides
        and stakes-based adjustment."""
        weights = dict(self._defaults)
        override = self._domain_overrides.get(task.task_class)
        if override:
            weights.update(override)
        weights = _apply_stakes_adjustment(weights, task.stakes)
        # Normalise to sum to 1.0 so downstream scoring is well defined.
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    # ------------------------------------------------------------------
    # Conflict resolution (§3.4)
    # ------------------------------------------------------------------
    def resolve(
        self,
        task: Task,
        candidates: dict[str, dict[str, float]],
        variance: Optional[dict[str, float]] = None,
        novelty: float = 0.0,
        hard_constraint_fn: Optional[Callable[[str], bool]] = None,
    ) -> ConflictResolution:
        """Pick a winning candidate given per-objective scores.

        1. Hard constraints (spec §3.4 step 1) are checked via
           `hard_constraint_fn`. A candidate that fails the constraint is
           dropped from consideration — safety overrides objective scores.
        2. Weighted objective score per candidate.
        3. If the top two candidates are within 5% of each other, prefer
           the lower-variance one (robustness tiebreaker).
        4. If novelty > 0.7, flag exploration budget allocated.
        """
        weights = self.weights_for(task)
        scored: dict[str, float] = {}
        for name, objective_scores in candidates.items():
            if hard_constraint_fn and not hard_constraint_fn(name):
                continue
            scored[name] = _weighted_score(objective_scores, weights)

        if not scored:
            raise ValueError("no candidates passed hard constraints")

        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        top_name, top_score = ranked[0]
        used_tiebreaker = False
        if len(ranked) >= 2:
            runner_name, runner_score = ranked[1]
            if top_score > 0 and (top_score - runner_score) / top_score < 0.05:
                # Within 5% — robustness tiebreaker
                if variance:
                    top_var = variance.get(top_name, 0.0)
                    runner_var = variance.get(runner_name, 0.0)
                    if runner_var < top_var:
                        top_name = runner_name
                        top_score = runner_score
                        used_tiebreaker = True

        exploration_allocated = novelty > 0.7

        return ConflictResolution(
            chosen=top_name,
            score=top_score,
            margin=(
                top_score - ranked[1][1] if len(ranked) > 1 else top_score
            ),
            used_tiebreaker=used_tiebreaker,
            exploration_allocated=exploration_allocated,
            runners_up=[(n, s) for n, s in ranked[1:]],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _apply_stakes_adjustment(
    weights: dict[str, float], stakes: Stakes
) -> dict[str, float]:
    """Spec §3.3 stakes-based adjustment."""
    w = dict(weights)
    if stakes == Stakes.LOW:
        w["latency"] = w.get("latency", 0.0) + 0.05
        # Reduce verification-aligned weights
        w["calibration"] = max(0.0, w.get("calibration", 0.0) - 0.05)
    elif stakes == Stakes.HIGH:
        w["truthfulness"] = w.get("truthfulness", 0.0) + 0.10
        w["calibration"] = w.get("calibration", 0.0) + 0.05
        w["latency"] = max(0.0, w.get("latency", 0.0) - 0.05)
    elif stakes == Stakes.CRITICAL:
        w["truthfulness"] = w.get("truthfulness", 0.0) + 0.20
        w["calibration"] = w.get("calibration", 0.0) + 0.10
        w["latency"] = 0.0  # "ignore latency" per §3.3
    # Medium stakes: no change.
    return w


def _weighted_score(
    objective_scores: dict[str, float], weights: dict[str, float]
) -> float:
    total = 0.0
    for key, w in weights.items():
        total += w * objective_scores.get(key, 0.0)
    return total
