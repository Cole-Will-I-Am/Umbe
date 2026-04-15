"""Telemetry Observer — AXIS v0.2 Priority 2 (spec §1.7).

The Observer is the **only** component permitted to write to the Self-Model
(§6 State Access Matrix). Every AxisRuntime.run() emits a TelemetryRecord
through the Observer, which:

    1. Stores the record in bounded ring buffers (per task class, per tool,
       per outcome — whatever the §1.7 derived-metrics table requires).
    2. Exposes `metrics()` to return the current derived metric snapshot.
    3. Exposes `self_model_snapshot()` to produce a telemetry-derived
       Self-Model dict that the Planner / Scheduler can read.
    4. Optionally tees every record to a durable JSONL sink so records
       survive process restarts and can be audited externally.

Fields from §1.7 that depend on later priorities (verifier_score,
confidence_predicted, routing_decision, user_correction) are optional.
Derived metrics that need those inputs degrade gracefully to `None`
when the data is missing rather than fabricating values — the Self-Model
is measured, never asserted.
"""

from __future__ import annotations

import json
import math
import statistics
import time
import uuid
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Iterable, Optional

from .types import ExecutionTrace, Plan, Strategy, Task, TaskOutcome

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Record type
# ---------------------------------------------------------------------------
@dataclass
class TelemetryRecord:
    """Per-task telemetry payload, matching the spec §1.7 schema."""

    task_id: str
    task_class: str
    strategy: Optional[str]
    outcome: str
    # Token accounting
    planning_tokens: int
    execution_tokens: int
    verification_tokens: int
    resource_cost_tokens: int
    # Timing
    total_latency_ms: int
    # Tooling
    tool_calls: list[dict]
    # Control-flow events
    replanning_events: int
    # Optional signals from later priorities
    routing_decision: Optional[dict] = None
    verifier_score: Optional[float] = None
    confidence_predicted: Optional[float] = None
    user_correction: Optional[bool] = None
    error: Optional[str] = None
    # Bookkeeping
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    recorded_at: float = field(default_factory=time.time)
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_trace(
        cls,
        trace: ExecutionTrace,
        task: Task,
        plan: Optional[Plan] = None,
    ) -> "TelemetryRecord":
        strategy_value: Optional[str]
        if trace.strategy is not None:
            strategy_value = trace.strategy.value
        elif plan is not None:
            strategy_value = plan.strategy.value
        else:
            strategy_value = None

        return cls(
            task_id=trace.task_id,
            task_class=trace.task_class or task.task_class,
            strategy=strategy_value,
            outcome=trace.outcome.value,
            planning_tokens=trace.planning_tokens,
            execution_tokens=trace.execution_tokens or trace.total_tokens,
            verification_tokens=trace.verification_tokens,
            resource_cost_tokens=(
                trace.planning_tokens
                + (trace.execution_tokens or trace.total_tokens)
                + trace.verification_tokens
            ),
            total_latency_ms=trace.latency_ms(),
            tool_calls=trace.tool_call_summary(),
            replanning_events=trace.replanning_events,
            routing_decision=trace.routing_decision,
            verifier_score=trace.verifier_score,
            confidence_predicted=trace.confidence_predicted,
            user_correction=trace.user_correction,
            error=trace.error,
        )


# ---------------------------------------------------------------------------
# Durable sink
# ---------------------------------------------------------------------------
class StructuredSink:
    """JSONL file sink. Append-only, line-delimited, schema-tagged."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: TelemetryRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.to_json() + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------
# Default rolling-window sizes from spec §1.7 derived-metrics table.
DEFAULT_WINDOWS: dict[str, int] = {
    "win_rate_per_class": 100,
    "calibration_error": 200,
    "strategy_bias": 500,
    "tool_success_rate": 100,
    "avg_cost_per_class": 100,
    "failure_mode_frequency": 100,
    "replanning_rate": 200,
    "latency_distribution": 500,
}


class TelemetryObserver:
    """Aggregates TelemetryRecords and exposes derived metrics + Self-Model.

    The Observer is the **sole** writer to the Self-Model (§6). All fields
    it emits are telemetry-derived; none are self-reported. Metrics that
    lack supporting data return `None` — we measure, we don't assert.
    """

    def __init__(
        self,
        sink: Optional[StructuredSink] = None,
        windows: Optional[dict[str, int]] = None,
        self_model_refresh_every: int = 50,
    ):
        self.sink = sink
        self.windows = {**DEFAULT_WINDOWS, **(windows or {})}
        self.self_model_refresh_every = self_model_refresh_every

        # ------------------------------------------------------------------
        # Ring buffers. Each is keyed where the spec table demands per-key
        # windows (per task class, per tool); otherwise a single global
        # deque.
        # ------------------------------------------------------------------
        self._tasks_total: int = 0
        self._records_all: Deque[TelemetryRecord] = deque(
            maxlen=self.windows["strategy_bias"]
        )
        self._outcomes_by_class: dict[str, Deque[str]] = {}
        self._costs_by_class: dict[str, Deque[int]] = {}
        self._latency_by_class: dict[str, Deque[int]] = {}
        self._replanning_events: Deque[int] = deque(
            maxlen=self.windows["replanning_rate"]
        )
        self._calibration_pairs: Deque[tuple[float, float]] = deque(
            maxlen=self.windows["calibration_error"]
        )
        self._strategy_selections: Deque[str] = deque(
            maxlen=self.windows["strategy_bias"]
        )
        self._failures: Deque[dict] = deque(
            maxlen=self.windows["failure_mode_frequency"]
        )
        self._tool_usage: dict[str, Deque[bool]] = {}

        # 2σ drift detection — tracks per-metric running statistics so we
        # can flag "any single metric moves more than 2 standard deviations
        # from its rolling mean" (spec §2, Self-Model update cadence).
        self._drift_flags: set[str] = set()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def record_trace(
        self,
        trace: ExecutionTrace,
        task: Task,
        plan: Optional[Plan] = None,
    ) -> TelemetryRecord:
        record = TelemetryRecord.from_trace(trace, task, plan)
        self.ingest(record)
        return record

    def ingest(self, record: TelemetryRecord) -> None:
        self._tasks_total += 1
        self._records_all.append(record)

        # Per-task-class outcome buffer
        cls = record.task_class
        self._outcomes_by_class.setdefault(
            cls, deque(maxlen=self.windows["win_rate_per_class"])
        ).append(record.outcome)

        # Per-task-class cost buffer
        self._costs_by_class.setdefault(
            cls, deque(maxlen=self.windows["avg_cost_per_class"])
        ).append(record.resource_cost_tokens)

        # Per-task-class latency buffer (p50/p90/p95/p99)
        self._latency_by_class.setdefault(
            cls, deque(maxlen=self.windows["latency_distribution"])
        ).append(record.total_latency_ms)

        # Replanning
        self._replanning_events.append(record.replanning_events)

        # Strategy bias
        if record.strategy:
            self._strategy_selections.append(record.strategy)

        # Calibration pairs (only when we have a predicted confidence)
        if record.confidence_predicted is not None:
            actual = 1.0 if record.outcome == TaskOutcome.SUCCESS.value else 0.0
            self._calibration_pairs.append((record.confidence_predicted, actual))

        # Failure clustering
        if record.outcome == TaskOutcome.FAILURE.value:
            self._failures.append(
                {
                    "task_class": cls,
                    "error": record.error or "unknown",
                    "strategy": record.strategy,
                }
            )

        # Tool success rate
        for tc in record.tool_calls:
            tool = tc.get("tool") or "unknown"
            self._tool_usage.setdefault(
                tool, deque(maxlen=self.windows["tool_success_rate"])
            ).append(bool(tc.get("success", True)))

        if self.sink is not None:
            self.sink.write(record)

    # ------------------------------------------------------------------
    # Derived metrics (spec §1.7 table)
    # ------------------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        return {
            "tasks_total": self._tasks_total,
            "win_rate_by_class": self._win_rate_by_class(),
            "confidence_calibration_error": self._calibration_error(),
            "strategy_bias": self._strategy_bias(),
            "tool_success_rate": self._tool_success_rate(),
            "avg_cost_by_class": self._avg_cost_by_class(),
            "failure_mode_frequency": self._failure_mode_frequency(),
            "replanning_rate": self._replanning_rate(),
            "latency_distribution_by_class": self._latency_distribution(),
        }

    def _win_rate_by_class(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for cls, outcomes in self._outcomes_by_class.items():
            if not outcomes:
                continue
            wins = sum(1 for o in outcomes if o == TaskOutcome.SUCCESS.value)
            out[cls] = wins / len(outcomes)
        return out

    def _calibration_error(self) -> Optional[float]:
        if not self._calibration_pairs:
            return None
        return sum(abs(p - a) for p, a in self._calibration_pairs) / len(
            self._calibration_pairs
        )

    def _strategy_bias(self) -> dict[str, float]:
        if not self._strategy_selections:
            return {}
        counts = Counter(self._strategy_selections)
        total = sum(counts.values())
        return {name: n / total for name, n in counts.items()}

    def _tool_success_rate(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for tool, results in self._tool_usage.items():
            if not results:
                continue
            out[tool] = sum(1 for r in results if r) / len(results)
        return out

    def _avg_cost_by_class(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for cls, costs in self._costs_by_class.items():
            if not costs:
                continue
            out[cls] = sum(costs) / len(costs)
        return out

    def _failure_mode_frequency(self) -> dict[str, float]:
        if not self._failures:
            return {}
        counts = Counter(f["error"] for f in self._failures)
        total = sum(counts.values())
        return {err: n / total for err, n in counts.items()}

    def _replanning_rate(self) -> Optional[float]:
        if not self._replanning_events:
            return None
        replanned = sum(1 for r in self._replanning_events if r > 0)
        return replanned / len(self._replanning_events)

    def _latency_distribution(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for cls, latencies in self._latency_by_class.items():
            if not latencies:
                continue
            sorted_lat = sorted(latencies)
            out[cls] = {
                "p50": _percentile(sorted_lat, 0.50),
                "p90": _percentile(sorted_lat, 0.90),
                "p95": _percentile(sorted_lat, 0.95),
                "p99": _percentile(sorted_lat, 0.99),
                "count": len(sorted_lat),
            }
        return out

    # ------------------------------------------------------------------
    # 2σ drift detector
    # ------------------------------------------------------------------
    def drift_flags(self, recent_n: int = 10, z_threshold: float = 2.0) -> set[str]:
        """Return names of metrics that have moved >2σ from their rolling baseline.

        Per spec §2: the Self-Model should be recomputed after every N tasks
        OR when any single metric moves more than 2 standard deviations from
        its rolling mean. We implement that as a two-sample test on the
        metric's ring buffer: split into `baseline` (everything except the
        tail) and `recent` (the last `recent_n` samples), and flag when the
        difference exceeds `z_threshold` standard errors.

        A two-sample test is used (rather than a 1-sample test against
        population stdev) because binary outcome series have the pathology
        that the stdev *grows* with the drift itself, masking it.
        """
        flags: set[str] = set()

        # Win rate per class: two-proportion z-test
        for cls, outcomes in self._outcomes_by_class.items():
            if len(outcomes) < 2 * recent_n:
                continue
            values = [
                1.0 if o == TaskOutcome.SUCCESS.value else 0.0 for o in outcomes
            ]
            baseline = values[:-recent_n]
            recent = values[-recent_n:]
            if _two_proportion_drift(baseline, recent, z_threshold):
                flags.add(f"win_rate:{cls}")

        # Latency per class: two-sample mean z-test (continuous data)
        for cls, latencies in self._latency_by_class.items():
            if len(latencies) < 2 * recent_n:
                continue
            values = [float(v) for v in latencies]
            baseline = values[:-recent_n]
            recent = values[-recent_n:]
            if _two_sample_mean_drift(baseline, recent, z_threshold):
                flags.add(f"latency:{cls}")

        return flags

    def should_refresh_self_model(self) -> bool:
        if self._tasks_total == 0:
            return False
        if self._tasks_total % self.self_model_refresh_every == 0:
            return True
        return bool(self.drift_flags())

    # ------------------------------------------------------------------
    # Self-Model writer (§6: only Telemetry Observer writes Self-Model)
    # ------------------------------------------------------------------
    def self_model_snapshot(self) -> dict[str, Any]:
        """Build a telemetry-derived Self-Model.

        Priority-2 scope: capabilities (win_rate, avg_cost_tokens),
        strategy_effectiveness from historical selections + outcomes,
        resource_profile (token averages, latency percentiles),
        failure_modes (from the failure cluster).

        Full P6 fields (best_strategy per class, confidence_calibration
        per class, known_failure_modes annotations) are deferred until
        the Verifier supplies predicted confidence.
        """
        metrics = self.metrics()

        # Per-class breakdown needs strategy + outcome + calibration joined.
        # We scan the whole record buffer once and fold.
        per_class_strategy_stats: dict[str, dict[str, dict[str, int]]] = {}
        per_class_calibration: dict[str, list[tuple[float, float]]] = {}
        per_class_failure_modes: dict[str, Counter] = {}
        for rec in self._records_all:
            cls = rec.task_class
            if rec.strategy:
                s = per_class_strategy_stats.setdefault(cls, {}).setdefault(
                    rec.strategy, {"total": 0, "wins": 0}
                )
                s["total"] += 1
                if rec.outcome == TaskOutcome.SUCCESS.value:
                    s["wins"] += 1
            if rec.confidence_predicted is not None:
                per_class_calibration.setdefault(cls, []).append(
                    (
                        rec.confidence_predicted,
                        1.0 if rec.outcome == TaskOutcome.SUCCESS.value else 0.0,
                    )
                )
            if rec.outcome == TaskOutcome.FAILURE.value:
                per_class_failure_modes.setdefault(cls, Counter())[
                    rec.error or "unknown"
                ] += 1

        capabilities: dict[str, dict[str, Any]] = {}
        for cls, win_rate in metrics["win_rate_by_class"].items():
            # Pick best_strategy as the one with the highest win rate for
            # this class, with a minimum sample size to avoid noise.
            best_strategy: Optional[str] = None
            best_rate = -1.0
            for strat, stats in per_class_strategy_stats.get(cls, {}).items():
                if stats["total"] < 3:
                    continue
                rate = stats["wins"] / stats["total"]
                if rate > best_rate:
                    best_rate = rate
                    best_strategy = strat

            # Per-class calibration error
            calib_pairs = per_class_calibration.get(cls, [])
            calibration_error: Optional[float] = None
            if calib_pairs:
                calibration_error = sum(abs(p - a) for p, a in calib_pairs) / len(
                    calib_pairs
                )

            known_modes = [
                m
                for m, _ in per_class_failure_modes.get(cls, Counter()).most_common(3)
            ]

            capabilities[cls] = {
                "win_rate": round(win_rate, 4),
                "avg_cost_tokens": round(
                    metrics["avg_cost_by_class"].get(cls, 0.0), 1
                ),
                "sample_size": len(self._outcomes_by_class[cls]),
                "best_strategy": best_strategy,
                "confidence_calibration_error": (
                    round(calibration_error, 4)
                    if calibration_error is not None
                    else None
                ),
                "known_failure_modes": known_modes,
            }

        # Strategy effectiveness = selection_rate × per-strategy win rate
        strategy_records: dict[str, list[float]] = {}
        strategy_wins: dict[str, int] = {}
        strategy_totals: dict[str, int] = {}
        for rec in self._records_all:
            if not rec.strategy:
                continue
            strategy_totals[rec.strategy] = strategy_totals.get(rec.strategy, 0) + 1
            if rec.outcome == TaskOutcome.SUCCESS.value:
                strategy_wins[rec.strategy] = strategy_wins.get(rec.strategy, 0) + 1
            strategy_records.setdefault(rec.strategy, []).append(
                rec.resource_cost_tokens
            )

        strategy_effectiveness: dict[str, dict[str, float]] = {}
        total_selections = sum(strategy_totals.values()) or 1
        for name, total in strategy_totals.items():
            wins = strategy_wins.get(name, 0)
            strategy_effectiveness[name] = {
                "selection_rate": round(total / total_selections, 4),
                "win_rate_when_selected": round(wins / total, 4) if total else 0.0,
                "sample_size": total,
            }

        # Resource profile — aggregate across all records we still hold
        latencies_all: list[int] = []
        token_costs: list[int] = []
        for rec in self._records_all:
            latencies_all.append(rec.total_latency_ms)
            token_costs.append(rec.resource_cost_tokens)
        resource_profile: dict[str, Any] = {}
        if token_costs:
            resource_profile["avg_tokens_per_task"] = round(
                sum(token_costs) / len(token_costs), 1
            )
        if latencies_all:
            sorted_lat = sorted(latencies_all)
            resource_profile["latency_p50_ms"] = _percentile(sorted_lat, 0.5)
            resource_profile["latency_p95_ms"] = _percentile(sorted_lat, 0.95)

        failure_modes = [
            {"pattern": err, "frequency": round(freq, 4)}
            for err, freq in metrics["failure_mode_frequency"].items()
        ]

        return {
            "capabilities": capabilities,
            "strategy_effectiveness": strategy_effectiveness,
            "resource_profile": resource_profile,
            "failure_modes": failure_modes,
            "last_updated": time.time(),
            "update_source": f"telemetry_observer:tasks={self._tasks_total}",
            "schema_version": SCHEMA_VERSION,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _two_proportion_drift(
    baseline: list[float], recent: list[float], z_threshold: float
) -> bool:
    """Two-proportion z-test for binary series.

    Returns True when the recent sample differs from the baseline by more
    than `z_threshold` pooled standard errors. Handles the degenerate
    zero-variance case (e.g. a pure-success baseline) by directly comparing
    means — any non-equal recent sample against a zero-variance baseline
    counts as drift.
    """
    if not baseline or not recent:
        return False
    n1, n2 = len(baseline), len(recent)
    p1 = sum(baseline) / n1
    p2 = sum(recent) / n2
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se_sq = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)
    if se_sq <= 0:
        return p1 != p2
    return abs(p2 - p1) / math.sqrt(se_sq) > z_threshold


def _two_sample_mean_drift(
    baseline: list[float], recent: list[float], z_threshold: float
) -> bool:
    """Two-sample mean z-test for continuous series."""
    if len(baseline) < 2 or len(recent) < 2:
        return False
    m1 = statistics.fmean(baseline)
    m2 = statistics.fmean(recent)
    v1 = statistics.pvariance(baseline)
    v2 = statistics.pvariance(recent)
    se_sq = v1 / len(baseline) + v2 / len(recent)
    if se_sq <= 0:
        return m1 != m2
    return abs(m2 - m1) / math.sqrt(se_sq) > z_threshold


def _percentile(sorted_values: list[float], q: float) -> float:
    """Return the q-quantile (0..1) from an already-sorted sequence.

    Uses linear interpolation between closest ranks. Empty input raises.
    """
    if not sorted_values:
        raise ValueError("empty input")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)
