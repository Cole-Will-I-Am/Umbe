"""Procedural Memory — spec §1.6.4.

Reusable execution patterns: workflows, decomposition templates, tool
routines, prompt templates. Write authority: Adaptation Manager only.
Procedures are created, versioned, and retired based on telemetry
evidence.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Procedure:
    procedure_id: str
    name: str
    task_class: str
    steps: list[dict]
    success_rate: float = 0.0
    sample_size: int = 0
    avg_cost: dict[str, float] = field(default_factory=dict)
    applicable_when: str = ""
    last_used: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    version: int = 1
    status: str = "candidate"  # candidate | validated | retired

    def to_dict(self) -> dict:
        return asdict(self)


class ProceduralMemory:
    """Per-task-class procedure registry with versioning and retirement."""

    def __init__(self) -> None:
        self._procedures: dict[str, Procedure] = {}

    # ------------------------------------------------------------------
    # Writes (Adaptation Manager only per §6)
    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        task_class: str,
        steps: list[dict],
        applicable_when: str = "",
    ) -> Procedure:
        pid = str(uuid.uuid4())
        proc = Procedure(
            procedure_id=pid,
            name=name,
            task_class=task_class,
            steps=list(steps),
            applicable_when=applicable_when,
        )
        self._procedures[pid] = proc
        return proc

    def update_steps(self, procedure_id: str, steps: list[dict]) -> Procedure:
        proc = self._procedures[procedure_id]
        proc.steps = list(steps)
        proc.version += 1
        proc.last_used = time.time()
        return proc

    def record_usage(
        self,
        procedure_id: str,
        success: bool,
        cost: Optional[dict[str, float]] = None,
    ) -> None:
        proc = self._procedures.get(procedure_id)
        if proc is None:
            return
        # Incremental success rate update
        n = proc.sample_size
        new_rate = (proc.success_rate * n + (1.0 if success else 0.0)) / (n + 1)
        proc.success_rate = new_rate
        proc.sample_size = n + 1
        proc.last_used = time.time()
        if cost:
            if not proc.avg_cost:
                proc.avg_cost = {k: float(v) for k, v in cost.items()}
            else:
                for k, v in cost.items():
                    prev = proc.avg_cost.get(k, 0.0)
                    proc.avg_cost[k] = (prev * n + float(v)) / (n + 1)

    def promote(self, procedure_id: str) -> None:
        proc = self._procedures.get(procedure_id)
        if proc:
            proc.status = "validated"

    def retire(self, procedure_id: str) -> None:
        proc = self._procedures.get(procedure_id)
        if proc:
            proc.status = "retired"

    def delete(self, procedure_id: str) -> None:
        self._procedures.pop(procedure_id, None)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._procedures)

    def get(self, procedure_id: str) -> Optional[Procedure]:
        return self._procedures.get(procedure_id)

    def for_class(
        self, task_class: str, include_candidates: bool = False
    ) -> list[Procedure]:
        valid_statuses = {"validated"}
        if include_candidates:
            valid_statuses.add("candidate")
        return [
            p
            for p in self._procedures.values()
            if p.task_class == task_class and p.status in valid_statuses
        ]

    def best_for_class(self, task_class: str) -> Optional[Procedure]:
        candidates = self.for_class(task_class, include_candidates=False)
        if not candidates:
            return None
        return max(candidates, key=lambda p: (p.success_rate, p.sample_size))

    def all(self) -> list[Procedure]:
        return list(self._procedures.values())
