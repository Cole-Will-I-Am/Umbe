"""Semantic Memory — spec §1.6.3.

Persistent, slowly-changing stable knowledge: user models, domain models,
system facts, distilled lessons. Write authority: MemoryManager only
(never written directly by Executor/Planner). Always derived from
accumulated episodes.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .embedding import cosine, embed


@dataclass
class SemanticEntity:
    entity_id: str
    type: str
    content: str
    confidence: float
    last_updated: float
    source_episodes: list[str] = field(default_factory=list)
    access_count: int = 0
    decay_score: float = 1.0
    embedding: list[float] = field(default_factory=list)
    version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


class SemanticMemory:
    """Content-keyed store of semantic entities."""

    def __init__(self) -> None:
        self._by_id: dict[str, SemanticEntity] = {}

    # ------------------------------------------------------------------
    # Writes (MemoryManager-only per §6)
    # ------------------------------------------------------------------
    def upsert(
        self,
        type: str,
        content: str,
        confidence: float,
        source_episodes: Optional[list[str]] = None,
        entity_id: Optional[str] = None,
    ) -> SemanticEntity:
        """Insert or update an entity. Version bumps on update."""
        source_episodes = source_episodes or []
        entity_id = entity_id or _entity_id_for(type, content)
        existing = self._by_id.get(entity_id)
        if existing is None:
            ent = SemanticEntity(
                entity_id=entity_id,
                type=type,
                content=content,
                confidence=confidence,
                last_updated=time.time(),
                source_episodes=list(source_episodes),
                embedding=embed(content),
                version=1,
            )
            self._by_id[entity_id] = ent
            return ent

        existing.content = content
        existing.confidence = confidence
        existing.source_episodes = list({*existing.source_episodes, *source_episodes})
        existing.last_updated = time.time()
        existing.embedding = embed(content)
        existing.version += 1
        return existing

    def archive(self, entity_id: str) -> None:
        self._by_id.pop(entity_id, None)

    def resolve_contradiction(
        self, entity_id_a: str, entity_id_b: str
    ) -> Optional[SemanticEntity]:
        """Spec §1.6.5: when two entries conflict, the one with more source
        episodes and higher confidence wins; loser is deleted. Returns the
        winner (or None if both were missing)."""
        a = self._by_id.get(entity_id_a)
        b = self._by_id.get(entity_id_b)
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        a_score = (len(a.source_episodes), a.confidence)
        b_score = (len(b.source_episodes), b.confidence)
        if a_score >= b_score:
            self.archive(b.entity_id)
            return a
        self.archive(a.entity_id)
        return b

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, entity_id: str) -> Optional[SemanticEntity]:
        ent = self._by_id.get(entity_id)
        if ent is not None:
            ent.access_count += 1
        return ent

    def all(self) -> list[SemanticEntity]:
        return list(self._by_id.values())

    def by_type(self, type: str) -> list[SemanticEntity]:
        return [e for e in self._by_id.values() if e.type == type]

    def similar(self, query: str, top_k: int = 10) -> list[tuple[SemanticEntity, float]]:
        q = embed(query)
        scored = [
            (e, cosine(q, e.embedding))
            for e in self._by_id.values()
            if e.embedding
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]


def _entity_id_for(type: str, content: str) -> str:
    # Stable id so re-distilling the same lesson upserts instead of dupes.
    import hashlib

    return f"{type}:{hashlib.blake2b(content.encode(), digest_size=8).hexdigest()}"
