"""Episodic Memory — spec §1.6.2.

One record per completed task. Retrieval by semantic similarity, task
class, recency, or outcome filter. Write authority: Telemetry Observer
(automated) and Adaptation Manager (postmortem annotations).
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Optional

from .embedding import cosine, embed


def _task_hash(task_input: str, task_class: str) -> str:
    h = hashlib.blake2b(
        f"{task_class}::{task_input}".encode(), digest_size=8
    ).hexdigest()
    return h


@dataclass
class Episode:
    """One completed-task record.

    Schema mirrors the §1.6.2 example. Fields that require signals from
    components that don't exist yet (calibration, verifier_score) are
    optional and default to None rather than fabricated values.
    """

    episode_id: str
    timestamp: float
    task_hash: str
    task_class: str
    input_summary: str
    strategy_used: Optional[str]
    tools_invoked: list[str]
    outcome: str
    outcome_detail: Optional[str] = None
    verifier_score: Optional[float] = None
    confidence_calibration: Optional[dict] = None
    resource_cost: dict[str, Any] = field(default_factory=dict)
    corrections: list[str] = field(default_factory=list)
    postmortem: Optional[str] = None
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class EpisodicMemory:
    """Per-instance append-only episode store with multi-key retrieval."""

    def __init__(self, max_episodes: int = 5_000):
        self._episodes: list[Episode] = []
        self._by_id: dict[str, int] = {}
        self.max_episodes = max_episodes

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def add(self, episode: Episode) -> None:
        if episode.episode_id in self._by_id:
            raise ValueError(f"duplicate episode_id: {episode.episode_id}")
        self._episodes.append(episode)
        self._by_id[episode.episode_id] = len(self._episodes) - 1
        # Hard cap eviction — oldest first. Forgetting Engine can apply
        # more sophisticated value-based eviction on top of this.
        while len(self._episodes) > self.max_episodes:
            evicted = self._episodes.pop(0)
            self._by_id.pop(evicted.episode_id, None)
            self._reindex()

    def annotate_postmortem(self, episode_id: str, postmortem: str) -> None:
        """Adaptation Manager write path."""
        idx = self._by_id.get(episode_id)
        if idx is None:
            raise KeyError(episode_id)
        self._episodes[idx].postmortem = postmortem

    def delete(self, episode_id: str) -> None:
        idx = self._by_id.pop(episode_id, None)
        if idx is None:
            return
        self._episodes.pop(idx)
        self._reindex()

    def _reindex(self) -> None:
        self._by_id = {e.episode_id: i for i, e in enumerate(self._episodes)}

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._episodes)

    def all(self) -> list[Episode]:
        return list(self._episodes)

    def by_class(self, task_class: str) -> list[Episode]:
        return [e for e in self._episodes if e.task_class == task_class]

    def by_outcome(self, outcome: str) -> list[Episode]:
        return [e for e in self._episodes if e.outcome == outcome]

    def recent(self, n: int = 50) -> list[Episode]:
        return list(self._episodes[-n:])

    def similar(
        self,
        query: str,
        top_k: int = 10,
        filter_fn: Optional[Callable[[Episode], bool]] = None,
    ) -> list[tuple[Episode, float]]:
        """Return the top-k episodes by cosine similarity to `query`."""
        q = embed(query)
        scored: list[tuple[Episode, float]] = []
        for ep in self._episodes:
            if filter_fn and not filter_fn(ep):
                continue
            if not ep.embedding:
                continue
            score = cosine(q, ep.embedding)
            scored.append((ep, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    def cluster_failures(
        self, task_class: Optional[str] = None
    ) -> dict[str, list[Episode]]:
        """Group failure episodes by their error string for postmortem input."""
        buckets: dict[str, list[Episode]] = {}
        for ep in self._episodes:
            if ep.outcome != "failure":
                continue
            if task_class and ep.task_class != task_class:
                continue
            key = (ep.outcome_detail or "unknown").strip() or "unknown"
            buckets.setdefault(key, []).append(ep)
        return buckets


def episode_from_trace(trace, task) -> Episode:
    """Helper to build an Episode from an ExecutionTrace + Task pair.

    Kept here (rather than on the trace class) so `axis.memory` has no
    upward dependency on `axis.telemetry`.
    """
    from ..types import ExecutionTrace, Task  # local import to avoid cycles

    assert isinstance(trace, ExecutionTrace) and isinstance(task, Task)
    tools = sorted({tc.get("tool") for sr in trace.step_results for tc in sr.tool_calls if tc.get("tool")})
    return Episode(
        episode_id=str(uuid.uuid4()),
        timestamp=time.time(),
        task_hash=_task_hash(task.input, task.task_class),
        task_class=task.task_class,
        input_summary=task.input[:256],
        strategy_used=trace.strategy.value if trace.strategy else None,
        tools_invoked=list(tools),
        outcome=trace.outcome.value,
        outcome_detail=trace.error,
        verifier_score=trace.verifier_score,
        confidence_calibration=(
            {"predicted": trace.confidence_predicted, "outcome": trace.outcome.value}
            if trace.confidence_predicted is not None
            else None
        ),
        resource_cost={
            "tokens": trace.total_tokens,
            "planning_tokens": trace.planning_tokens,
            "verification_tokens": trace.verification_tokens,
            "latency_ms": trace.latency_ms(),
        },
        embedding=embed(task.input),
    )
