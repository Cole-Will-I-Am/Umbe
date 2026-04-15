"""Memory Manager — the single entry point to all persistent memory stores.

Enforces spec §6 write-authority rules:

    Executor          → Working Memory only
    Telemetry Observer → Episodic Memory (automated writes)
    Memory Manager    → Semantic Memory (distillation only)
    Adaptation Manager → Procedural Memory; Episodic postmortem annotations

The Manager exposes narrow methods for the authorised writers. Reads are
open to Planner/Verifier/Policy Router per §6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .episodic import Episode, EpisodicMemory, episode_from_trace
from .forgetting import ForgettingEngine, ForgettingPolicy, ForgettingReport
from .procedural import Procedure, ProceduralMemory
from .semantic import SemanticEntity, SemanticMemory

if TYPE_CHECKING:
    from ..types import ExecutionTrace, Task


class MemoryManager:
    def __init__(
        self,
        episodic: Optional[EpisodicMemory] = None,
        semantic: Optional[SemanticMemory] = None,
        procedural: Optional[ProceduralMemory] = None,
        forgetting: Optional[ForgettingEngine] = None,
    ) -> None:
        self.episodic = episodic or EpisodicMemory()
        self.semantic = semantic or SemanticMemory()
        self.procedural = procedural or ProceduralMemory()
        self.forgetting = forgetting or ForgettingEngine()

    # ------------------------------------------------------------------
    # Telemetry Observer write path
    # ------------------------------------------------------------------
    def record_episode(self, trace: "ExecutionTrace", task: "Task") -> Episode:
        """Called by the runtime after each task completes."""
        ep = episode_from_trace(trace, task)
        self.episodic.add(ep)
        return ep

    # ------------------------------------------------------------------
    # Memory Manager distillation path
    # ------------------------------------------------------------------
    def distill(self, task_class: str) -> int:
        return self.forgetting.distill_similar_episodes(
            self.episodic, self.semantic, task_class
        )

    def forget_cycle(self) -> ForgettingReport:
        return self.forgetting.run(self.episodic, self.semantic, self.procedural)

    # ------------------------------------------------------------------
    # Adaptation Manager write paths
    # ------------------------------------------------------------------
    def annotate_episode(self, episode_id: str, postmortem: str) -> None:
        self.episodic.annotate_postmortem(episode_id, postmortem)

    def register_procedure(
        self,
        name: str,
        task_class: str,
        steps: list[dict],
        applicable_when: str = "",
    ) -> Procedure:
        return self.procedural.register(name, task_class, steps, applicable_when)

    def record_procedure_usage(
        self,
        procedure_id: str,
        success: bool,
        cost: Optional[dict] = None,
    ) -> None:
        self.procedural.record_usage(procedure_id, success, cost)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def procedural_index_for_planner(self) -> dict[str, dict]:
        """Return best procedures per task class, formatted so the Planner
        can consume them directly as its procedural_memory dict."""
        index: dict[str, dict] = {}
        for proc in self.procedural.all():
            if proc.status != "validated":
                continue
            best = index.get(proc.task_class)
            if best is None or proc.success_rate > best.get("success_rate", 0.0):
                index[proc.task_class] = {
                    "name": proc.name,
                    "procedure_id": proc.procedure_id,
                    "success_rate": proc.success_rate,
                    "steps": proc.steps,
                }
        return index
