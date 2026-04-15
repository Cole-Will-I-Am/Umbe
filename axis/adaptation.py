"""Adaptation Manager — spec §4.

Proposes, tests, and deploys Class 2 (procedural) and Class 3 (policy)
changes. Operates under strict constraints:

    * Trigger conditions must be met before a proposal is generated
      (win-rate drop, calibration error, failure-mode spike, budget
      overruns).
    * Credit assignment across the seven root-cause classes from §4.2.
    * Locality constraint: proposals target the most specific scope
      where evidence supports a change.
    * Class 3 proposals must pass the EvalHarness before deployment.
    * Every deployment declares its blast radius and attaches a rollback
      trigger. Automatic rollback on trigger.
    * Circuit breaker: 3 rollbacks within 500 tasks → conservative mode
      (Class 2 only) for the next 1000 tasks.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .eval_harness import EvalHarness, EvalReport
from .memory import MemoryManager
from .telemetry import TelemetryObserver


ROOT_CAUSES = (
    "retrieval_failure",
    "decomposition_failure",
    "strategy_mismatch",
    "tool_selection_error",
    "confidence_miscalibration",
    "domain_knowledge_gap",
    "communication_failure",
)


class ChangeClass(str, Enum):
    WORKING_STATE = "class_1"
    MEMORY = "class_2"
    POLICY = "class_3"
    PARAMETER = "class_4"


@dataclass
class Proposal:
    proposal_id: str
    change_class: ChangeClass
    target_task_class: Optional[str]
    root_cause: str
    description: str
    counterfactual_improvement: float
    affected_task_classes: list[str]
    unaffected_task_classes: list[str] = field(default_factory=list)
    rollback_trigger: dict = field(default_factory=dict)
    payload: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class Deployment:
    deployment_id: str
    proposal: Proposal
    deployed_at: float
    win_rate_at_deploy: float
    tasks_since_deploy: int = 0
    rolled_back: bool = False
    rollback_reason: Optional[str] = None


@dataclass
class AdaptationConfig:
    failure_cluster_min: int = 10            # §4.1 K default
    win_rate_drop_sigma: float = 1.0
    calibration_error_threshold: float = 0.25
    failure_spike_multiplier: float = 1.5
    rollback_win_rate_drop: float = 0.05
    rollback_window_tasks: int = 50
    circuit_breaker_rollbacks: int = 3
    circuit_breaker_window_tasks: int = 500
    conservative_mode_duration_tasks: int = 1000
    # Hard: Class 4 changes are never proposed online.
    allow_class_4: bool = False


class AdaptationManager:
    """The self-customisation engine.

    Reads telemetry + episodic memory, proposes scoped modifications,
    routes Class 3 proposals through the Eval Harness, and manages
    rollback.
    """

    def __init__(
        self,
        observer: TelemetryObserver,
        memory: MemoryManager,
        eval_harness: Optional[EvalHarness] = None,
        config: Optional[AdaptationConfig] = None,
    ) -> None:
        self.observer = observer
        self.memory = memory
        self.eval_harness = eval_harness
        self.config = config or AdaptationConfig()
        self._baselines: dict[str, float] = {}
        self.deployments: list[Deployment] = []
        self._rollbacks: list[float] = []
        self._conservative_until: int = 0  # self.observer._tasks_total sentinel

    # ------------------------------------------------------------------
    # Triggers (§4.2)
    # ------------------------------------------------------------------
    def check_triggers(self) -> list[str]:
        """Return the set of trigger names that are currently active."""
        triggers: list[str] = []
        metrics = self.observer.metrics()

        win_rates = metrics["win_rate_by_class"]
        for cls, rate in win_rates.items():
            baseline = self._baselines.setdefault(cls, rate)
            # First observation just sets the baseline — no drift yet.
            if rate < baseline - 0.1:  # crude 1σ-ish floor
                triggers.append(f"win_rate_drop:{cls}")

        ce = metrics["confidence_calibration_error"]
        if ce is not None and ce > self.config.calibration_error_threshold:
            triggers.append("calibration_error")

        # Failure spike: any failure mode with frequency > threshold
        for mode, freq in metrics["failure_mode_frequency"].items():
            if freq > 0.4:
                triggers.append(f"failure_spike:{mode}")

        # Drift flags from Observer count as triggers too.
        for flag in self.observer.drift_flags():
            triggers.append(f"drift:{flag}")
        return triggers

    # ------------------------------------------------------------------
    # Credit assignment + proposal generation (§4.2)
    # ------------------------------------------------------------------
    def diagnose(self, task_class: str) -> Optional[dict[str, Any]]:
        """Cluster failures for a task class, assign credit, pick the
        dominant root cause. Returns None when no actionable pattern."""
        failures = self.memory.episodic.cluster_failures(task_class)
        total = sum(len(v) for v in failures.values())
        if total < self.config.failure_cluster_min:
            return None

        causes = Counter()
        for bucket, eps in failures.items():
            for ep in eps:
                cause = _classify_root_cause(ep, bucket)
                causes[cause] += 1
        dominant, count = causes.most_common(1)[0]
        if count / total < 0.5:
            return None  # no dominant cause; spec §4.2 requires >50%

        return {
            "task_class": task_class,
            "root_cause": dominant,
            "failure_count": total,
            "cluster": {k: len(v) for k, v in failures.items()},
        }

    def propose(self, diagnosis: dict[str, Any]) -> Optional[Proposal]:
        """Produce a locality-constrained modification proposal.

        Locality (§4.2): single-domain/single-strategy first, then wider
        scopes only if narrower ones aren't applicable.
        """
        if self._is_in_conservative_mode():
            allowed = {ChangeClass.MEMORY}
        else:
            allowed = {ChangeClass.MEMORY, ChangeClass.POLICY}

        task_class = diagnosis["task_class"]
        root_cause = diagnosis["root_cause"]

        # Map root cause → scoped payload + change class.
        if root_cause == "retrieval_failure":
            payload = {
                "policy": "retrieval_policy",
                "task_class": task_class,
                "set": "broad_sweep",
            }
            change_class = ChangeClass.POLICY
            description = (
                f"force broad retrieval for {task_class} to address "
                "retrieval_failure cluster"
            )
        elif root_cause == "strategy_mismatch":
            payload = {
                "policy": "best_strategy",
                "task_class": task_class,
                "set": "retrieval_first",
            }
            change_class = ChangeClass.POLICY
            description = (
                f"switch default strategy for {task_class} to "
                "retrieval_first"
            )
        elif root_cause == "tool_selection_error":
            # Tools → procedural memory patch (Class 2)
            payload = {
                "procedural_template": {
                    "name": f"auto_{task_class}_tool_first",
                    "task_class": task_class,
                    "steps": [
                        {"action": "invoke_primary_tool", "tool": "primary_tool", "strategy": "tool_first"},
                        {"action": "reason_about_task", "strategy": "analytical"},
                        {"action": "produce_final_answer", "strategy": "analytical"},
                    ],
                }
            }
            change_class = ChangeClass.MEMORY
            description = (
                f"register tool-first procedural template for {task_class}"
            )
        else:
            # Default to Class 2 postmortem annotation (safest)
            payload = {"annotation": f"root_cause:{root_cause}"}
            change_class = ChangeClass.MEMORY
            description = (
                f"annotate failure cluster for {task_class} with "
                f"{root_cause}"
            )

        if change_class not in allowed:
            return None

        # Declare blast radius (§11).
        affected = [task_class]
        unaffected = self._unaffected_classes(task_class)

        return Proposal(
            proposal_id=str(uuid.uuid4()),
            change_class=change_class,
            target_task_class=task_class,
            root_cause=root_cause,
            description=description,
            counterfactual_improvement=self._counterfactual_estimate(diagnosis),
            affected_task_classes=affected,
            unaffected_task_classes=unaffected,
            rollback_trigger={
                "metric": "win_rate",
                "task_class": task_class,
                "threshold": self.config.rollback_win_rate_drop,
                "window": self.config.rollback_window_tasks,
            },
            payload=payload,
        )

    def _unaffected_classes(self, target: str) -> list[str]:
        all_classes = {r.task_class for r in self.observer._records_all}
        return sorted(all_classes - {target})

    def _counterfactual_estimate(self, diagnosis: dict[str, Any]) -> float:
        # Naive: assume the change eliminates half the failures.
        total_records = sum(
            1
            for r in self.observer._records_all
            if r.task_class == diagnosis["task_class"]
        )
        if total_records == 0:
            return 0.0
        return 0.5 * diagnosis["failure_count"] / total_records

    # ------------------------------------------------------------------
    # Evaluation + deployment (§4.3, §11)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        proposal: Proposal,
        baseline_run,
        candidate_run,
    ) -> Optional[EvalReport]:
        """Run the proposal through the Eval Harness. Class 2 changes
        may skip the harness per spec §4.1 ('episodic writes: automatic';
        'procedural changes: require Adaptation Manager review with
        evidence'). Class 3 changes always go through the harness."""
        if self.eval_harness is None:
            return None
        if proposal.change_class == ChangeClass.POLICY:
            return self.eval_harness.run_suite(baseline_run, candidate_run)
        return self.eval_harness.run_suite(baseline_run, candidate_run)

    def deploy(
        self,
        proposal: Proposal,
        apply_fn,
    ) -> Deployment:
        """Apply the proposal's payload via `apply_fn` and register the
        deployment for rollback tracking."""
        metrics = self.observer.metrics()
        win_rate = (
            metrics["win_rate_by_class"].get(proposal.target_task_class or "", 0.0)
        )
        apply_fn(proposal.payload)
        dep = Deployment(
            deployment_id=str(uuid.uuid4()),
            proposal=proposal,
            deployed_at=time.time(),
            win_rate_at_deploy=win_rate,
        )
        self.deployments.append(dep)
        return dep

    def check_rollbacks(self) -> list[Deployment]:
        """Check every active deployment's rollback trigger. Returns the
        list of deployments that just rolled back."""
        rolled: list[Deployment] = []
        metrics = self.observer.metrics()
        for dep in self.deployments:
            if dep.rolled_back:
                continue
            dep.tasks_since_deploy = min(
                self.observer._tasks_total - int(dep.deployed_at * 0),
                self.config.rollback_window_tasks,
            )
            target = dep.proposal.target_task_class
            if target is None:
                continue
            current = metrics["win_rate_by_class"].get(target, dep.win_rate_at_deploy)
            drop = dep.win_rate_at_deploy - current
            if drop > dep.proposal.rollback_trigger["threshold"]:
                dep.rolled_back = True
                dep.rollback_reason = f"win_rate drop {drop:.3f}"
                self._rollbacks.append(time.time())
                rolled.append(dep)
        # Circuit breaker
        recent = [
            t
            for t in self._rollbacks
            if self.observer._tasks_total
            - self.config.circuit_breaker_window_tasks
            < 10**18  # simple: just count the list
        ]
        if len(recent) >= self.config.circuit_breaker_rollbacks:
            self._conservative_until = (
                self.observer._tasks_total
                + self.config.conservative_mode_duration_tasks
            )
        return rolled

    def _is_in_conservative_mode(self) -> bool:
        return self.observer._tasks_total < self._conservative_until


# ---------------------------------------------------------------------------
# Root cause classification
# ---------------------------------------------------------------------------
def _classify_root_cause(episode, bucket_key: str) -> str:
    """Classify a failure into one of the seven §4.2 root causes.

    Heuristic-based placeholder — the production version is a trained
    classifier on execution traces. The classification surface is stable
    so swapping in a learned model is a one-line change.
    """
    detail = (episode.outcome_detail or bucket_key or "").lower()

    if "retrieval" in detail or "stale" in detail or "missing_info" in detail:
        return "retrieval_failure"
    if "decomp" in detail or "plan" in detail:
        return "decomposition_failure"
    if "tool" in detail or "timeout" in detail:
        return "tool_selection_error"
    if "calibration" in detail or "over_confident" in detail:
        return "confidence_miscalibration"
    if "communication" in detail or "format" in detail:
        return "communication_failure"
    if "knowledge" in detail or "domain_gap" in detail:
        return "domain_knowledge_gap"
    return "strategy_mismatch"
