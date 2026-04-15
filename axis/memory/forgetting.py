"""Forgetting Engine — spec §1.6.5.

Memory without forgetting is memory bloat. Applies six mechanisms:

    recency decay           — old unused episodic entries are compressed
    relevance pruning       — semantic entities with low decay_score archived
    frequency gating        — stale low-win-rate procedures retired
    contradiction resolution — handled at the semantic store (winner-takes-all)
    compression             — clusters of similar episodes distilled to semantic
    budget enforcement      — hard cap on total entries via value function
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .episodic import Episode, EpisodicMemory
    from .procedural import ProceduralMemory
    from .semantic import SemanticMemory


@dataclass
class ForgettingPolicy:
    # Recency decay
    episodic_stale_age_s: float = 30 * 24 * 60 * 60  # 30 days
    # Relevance pruning
    semantic_decay_floor: float = 0.1
    # Frequency gating
    procedural_min_success_rate: float = 0.5
    procedural_idle_retire_s: float = 14 * 24 * 60 * 60  # 14 days
    # Budget enforcement
    episodic_max: int = 5_000
    semantic_max: int = 2_000
    procedural_max: int = 500
    # Value function weights
    recency_weight: float = 0.25
    frequency_weight: float = 0.25
    utility_weight: float = 0.35
    uniqueness_weight: float = 0.15


@dataclass
class ForgettingReport:
    episodic_compressed: int = 0
    episodic_evicted: int = 0
    semantic_archived: int = 0
    procedural_retired: int = 0
    distilled_entities: int = 0


class ForgettingEngine:
    def __init__(self, policy: Optional[ForgettingPolicy] = None):
        self.policy = policy or ForgettingPolicy()

    def run(
        self,
        episodic: "EpisodicMemory",
        semantic: "SemanticMemory",
        procedural: "ProceduralMemory",
        now: Optional[float] = None,
    ) -> ForgettingReport:
        now = now if now is not None else time.time()
        report = ForgettingReport()

        # 1. Episodic compression — stale + zero retrieval hits (approximated
        # here by age only; when Memory Manager tracks retrieval hits we'll
        # feed those in).
        for ep in list(episodic.all()):
            if now - ep.timestamp > self.policy.episodic_stale_age_s:
                # Compress by summarising in-place and archiving the
                # original. For Priority-1 scope this means dropping the
                # embedding and truncating the input summary.
                ep.embedding = []
                ep.input_summary = ep.input_summary[:64]
                report.episodic_compressed += 1

        # 2. Semantic relevance pruning.
        for ent in list(semantic.all()):
            if ent.decay_score < self.policy.semantic_decay_floor:
                semantic.archive(ent.entity_id)
                report.semantic_archived += 1

        # 3. Procedural frequency gating.
        for proc in list(procedural.all()):
            stale = (now - proc.last_used) > self.policy.procedural_idle_retire_s
            weak = proc.sample_size > 0 and proc.success_rate < self.policy.procedural_min_success_rate
            if proc.status != "retired" and (weak and stale):
                procedural.retire(proc.procedure_id)
                report.procedural_retired += 1

        # 4. Budget enforcement — value-function eviction.
        if len(episodic) > self.policy.episodic_max:
            excess = len(episodic) - self.policy.episodic_max
            scored = sorted(
                episodic.all(),
                key=lambda e: self._episode_value(e, now),
            )
            for ep in scored[:excess]:
                episodic.delete(ep.episode_id)
                report.episodic_evicted += 1

        return report

    def distill_similar_episodes(
        self,
        episodic: "EpisodicMemory",
        semantic: "SemanticMemory",
        task_class: str,
        min_cluster_size: int = 3,
    ) -> int:
        """Cluster successful episodes in a task class and write one
        distilled semantic lesson per cluster. Returns the count of new
        or updated semantic entities."""
        successes = [
            e for e in episodic.by_class(task_class) if e.outcome == "success"
        ]
        if len(successes) < min_cluster_size:
            return 0

        # Naive clustering: group by strategy_used.
        clusters: dict[str, list] = {}
        for ep in successes:
            key = ep.strategy_used or "unknown"
            clusters.setdefault(key, []).append(ep)

        count = 0
        for strategy, eps in clusters.items():
            if len(eps) < min_cluster_size:
                continue
            content = (
                f"task_class={task_class} strategy={strategy} "
                f"success_rate=1.0 sample_size={len(eps)} "
                f"distilled from {len(eps)} successful episodes"
            )
            ent = semantic.upsert(
                type="distilled_lesson",
                content=content,
                confidence=min(1.0, 0.5 + 0.1 * len(eps)),
                source_episodes=[e.episode_id for e in eps],
            )
            count += 1
        return count

    def _episode_value(self, ep, now: float) -> float:
        """Spec §1.6.5 value function. Higher = keep."""
        age_s = max(1.0, now - ep.timestamp)
        # Normalise recency into [0,1] — most-recent is 1.0
        recency = 1.0 / (1.0 + age_s / 86_400.0)
        frequency = 0.0  # Memory Manager will start tracking hits in P7+
        utility = 1.0 if ep.outcome == "success" else 0.4
        uniqueness = 1.0  # stub — Memory Manager can compute cluster density
        p = self.policy
        return (
            p.recency_weight * recency
            + p.frequency_weight * frequency
            + p.utility_weight * utility
            + p.uniqueness_weight * uniqueness
        )
