"""Multi-instance coordination — spec §10.

Three models from §10.1:

    A — centralized policy, local execution
    B — federated family (recommended default, §10.2)
    C — swarm with shared procedural core

We implement the federated (Option B) coordinator. Each instance keeps
local adapters and episodic memory; semantic + procedural memory are
shared; the Self-Model is computed from aggregated telemetry.

Procedural merge rules (§10.2):
    * A procedure proposed by one instance is tagged `candidate`.
    * It becomes `validated` when 3+ instances report success rate
      above threshold on the same task class.
    * Validated procedures replace or supplement existing ones.
    * Local overrides are permitted but logged and reviewed.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .memory import ProceduralMemory
from .telemetry import TelemetryObserver


@dataclass
class InstanceReport:
    instance_id: str
    task_class: str
    procedure_id: str
    success_rate: float
    sample_size: int
    reported_at: float = field(default_factory=time.time)


@dataclass
class LocalOverride:
    instance_id: str
    procedure_id: str
    reason: str
    created_at: float = field(default_factory=time.time)


class FederatedCoordinator:
    """Option B federated coordinator.

    Maintains a shared procedural memory and a map of per-instance reports.
    """

    def __init__(
        self,
        validation_threshold: float = 0.7,
        min_instances_to_validate: int = 3,
    ) -> None:
        self._shared_procedural = ProceduralMemory()
        self._reports: dict[str, list[InstanceReport]] = defaultdict(list)
        self._local_overrides: list[LocalOverride] = []
        self._registered_instances: set[str] = set()
        self.validation_threshold = validation_threshold
        self.min_instances_to_validate = min_instances_to_validate

    # ------------------------------------------------------------------
    # Instance registration
    # ------------------------------------------------------------------
    def register_instance(self, instance_id: str) -> None:
        self._registered_instances.add(instance_id)

    def instance_count(self) -> int:
        return len(self._registered_instances)

    # ------------------------------------------------------------------
    # Procedural memory sharing
    # ------------------------------------------------------------------
    @property
    def shared_procedural(self) -> ProceduralMemory:
        return self._shared_procedural

    def propose_candidate(
        self,
        instance_id: str,
        name: str,
        task_class: str,
        steps: list[dict],
        applicable_when: str = "",
    ):
        self.register_instance(instance_id)
        proc = self._shared_procedural.register(
            name=name,
            task_class=task_class,
            steps=steps,
            applicable_when=applicable_when,
        )
        return proc

    def report_usage(
        self,
        instance_id: str,
        procedure_id: str,
        success_rate: float,
        sample_size: int,
        task_class: Optional[str] = None,
    ) -> None:
        self.register_instance(instance_id)
        tc = task_class
        if tc is None:
            proc = self._shared_procedural.get(procedure_id)
            if proc is None:
                return
            tc = proc.task_class
        self._reports[procedure_id].append(
            InstanceReport(
                instance_id=instance_id,
                task_class=tc,
                procedure_id=procedure_id,
                success_rate=success_rate,
                sample_size=sample_size,
            )
        )

    def check_validations(self) -> list:
        """Promote candidate procedures that meet the merge rules."""
        promoted = []
        for pid, reports in self._reports.items():
            proc = self._shared_procedural.get(pid)
            if proc is None or proc.status != "candidate":
                continue
            # Count distinct instances meeting the threshold
            distinct_good = {
                r.instance_id
                for r in reports
                if r.success_rate > self.validation_threshold
            }
            if len(distinct_good) >= self.min_instances_to_validate:
                self._shared_procedural.promote(pid)
                promoted.append(proc)
        return promoted

    def record_local_override(
        self, instance_id: str, procedure_id: str, reason: str
    ) -> LocalOverride:
        override = LocalOverride(
            instance_id=instance_id,
            procedure_id=procedure_id,
            reason=reason,
        )
        self._local_overrides.append(override)
        return override

    def overrides_for_review(self) -> list[LocalOverride]:
        return list(self._local_overrides)

    # ------------------------------------------------------------------
    # Aggregated Self-Model from distributed telemetry
    # ------------------------------------------------------------------
    def aggregate_self_model(
        self, observers: dict[str, TelemetryObserver]
    ) -> dict:
        """Merge per-instance telemetry observers into one Self-Model.

        Simple weighted-by-sample-size average of capability win rates.
        Losing any single instance does not corrupt the Self-Model —
        other instances keep reporting.
        """
        capabilities: dict[str, list[tuple[float, int]]] = defaultdict(list)
        failure_modes: Counter = Counter()

        for observer in observers.values():
            snap = observer.self_model_snapshot()
            for cls, cap in snap.get("capabilities", {}).items():
                win_rate = cap.get("win_rate")
                sample = cap.get("sample_size", 0)
                if win_rate is not None and sample > 0:
                    capabilities[cls].append((win_rate, sample))
                for fm in cap.get("known_failure_modes", []):
                    failure_modes[fm] += 1

        merged_capabilities: dict = {}
        for cls, pairs in capabilities.items():
            total = sum(s for _, s in pairs)
            weighted = sum(r * s for r, s in pairs) / total if total else 0.0
            merged_capabilities[cls] = {
                "win_rate": round(weighted, 4),
                "sample_size": total,
                "contributing_instances": len(pairs),
            }

        return {
            "capabilities": merged_capabilities,
            "failure_modes": [
                {"pattern": fm, "frequency": count}
                for fm, count in failure_modes.most_common()
            ],
            "instances": len(observers),
            "last_aggregated": time.time(),
        }
