"""Reference Frame operational variables — spec §9.

Six variables that feed specific control-flow decisions in the
Scheduler and Policy Router. From spec §9 Operational:

    ComputationPosition   → stop-condition / replan trigger
    KnowledgeTopology     → retrieval / verification depth
    ArchitectureAwareness → plan feasibility (context window, tools)
    ResourceHorizon       → prioritisation / compression
    InteractionModel      → response posture
    GradientSignals       → verify / retrieve / replan / escalate gates

Each variable is a plain structured dataclass. The point is that the
system can address "where am I right now" without inventing the answer
— every field is measured or configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ComputationPosition:
    """Where we are in the current task's compute budget."""

    tokens_used: int
    tokens_remaining: int
    planning_tokens_spent: int = 0
    verification_tokens_spent: int = 0
    replanning_events: int = 0

    @property
    def budget_fraction_spent(self) -> float:
        total = self.tokens_used + self.tokens_remaining
        return self.tokens_used / total if total > 0 else 0.0

    def should_force_stop(self, marginal_gain_estimate: float) -> bool:
        """True when the Scheduler should commit the current best answer."""
        if self.tokens_remaining <= 0:
            return True
        # Diminishing returns: if we've burned >80% of the budget and
        # marginal gain is small, stop.
        if self.budget_fraction_spent > 0.8 and marginal_gain_estimate < 0.1:
            return True
        return False


@dataclass
class KnowledgeTopology:
    """How confident we are in this domain, and how far from the frontier."""

    domain: str
    domain_confidence: float
    frontier_distance: float  # 0 = center of known territory, 1 = frontier
    known_failure_modes: list[str] = field(default_factory=list)

    def retrieval_depth(self) -> str:
        if self.domain_confidence > 0.85 and self.frontier_distance < 0.2:
            return "none"
        if self.frontier_distance > 0.6 or self.domain_confidence < 0.5:
            return "broad_sweep"
        return "targeted"

    def verification_depth(self) -> str:
        if self.domain_confidence > 0.9:
            return "spot_check"
        if self.domain_confidence < 0.6 or self.frontier_distance > 0.7:
            return "full"
        return "spot_check"


@dataclass
class ArchitectureAwareness:
    """What the runtime can physically do right now."""

    context_window_tokens: int
    context_tokens_used: int
    tools_available: list[str]
    max_concurrent_tool_calls: int = 1
    modalities: list[str] = field(default_factory=lambda: ["text"])

    @property
    def context_remaining(self) -> int:
        return max(0, self.context_window_tokens - self.context_tokens_used)

    def plan_is_feasible(
        self, projected_context_tokens: int, required_tools: list[str]
    ) -> bool:
        if projected_context_tokens > self.context_remaining:
            return False
        for t in required_tools:
            if t not in self.tools_available:
                return False
        return True


@dataclass
class ResourceHorizon:
    """What we can still afford to do and what to cut first."""

    remaining_task_tokens: int
    session_tokens_remaining: int
    wall_clock_remaining_ms: Optional[int] = None

    def must_prioritise(self) -> bool:
        return self.remaining_task_tokens < 500 or (
            self.wall_clock_remaining_ms is not None
            and self.wall_clock_remaining_ms < 2000
        )

    def should_compress(self) -> bool:
        return self.session_tokens_remaining < 2000


@dataclass
class InteractionModel:
    """How the response should be framed for the current collaborator/channel."""

    stakes: str  # low | medium | high | critical
    collaborator_expertise: str = "unknown"  # novice | peer | expert | unknown
    channel: str = "text"  # text | voice | structured_api

    def response_posture(self) -> str:
        if self.stakes in ("high", "critical"):
            return "cautious"
        if self.collaborator_expertise == "novice":
            return "thorough"
        if self.channel == "voice":
            return "concise"
        return "thorough"


@dataclass
class GradientSignals:
    """Local signals that drive gate decisions (§9)."""

    confidence: float
    novelty: float
    coherence: float
    friction: int

    def should_retrieve(self) -> bool:
        return self.novelty > 0.4 or self.confidence < 0.6

    def should_verify(self) -> bool:
        return self.confidence < 0.7 or self.coherence < 0.8

    def should_replan(self) -> bool:
        return self.friction >= 2 or self.confidence < 0.4

    def should_escalate(self) -> bool:
        return self.friction >= 4 and self.confidence < 0.5


@dataclass
class ReferenceFrame:
    """Aggregate of all six operational variables at a single instant."""

    computation_position: ComputationPosition
    knowledge_topology: KnowledgeTopology
    architecture_awareness: ArchitectureAwareness
    resource_horizon: ResourceHorizon
    interaction_model: InteractionModel
    gradient_signals: GradientSignals
