"""The InferenceBackend protocol and its result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


@dataclass
class BackendResult:
    """What an InferenceBackend.run() call returns.

    `entropy` is the spec's confidence signal source (token entropy +
    trajectory disagreement). The stub backend returns a fixed value; a
    real backend should return a calibrated estimate for the Verifier
    (Priority 4) to consume.

    `cache_read_tokens` / `cache_creation_tokens` report prompt caching
    usage when the backend supports it. Both default to 0 for backends
    that don't cache.
    """

    output: str
    tokens_used: int
    success: bool = True
    error: Optional[str] = None
    entropy: float = 0.0
    tool_calls: list[dict] = field(default_factory=list)
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@runtime_checkable
class InferenceBackend(Protocol):
    """Minimal interface the Executor requires from an inference backend."""

    def run(
        self,
        prompt: str,
        max_tokens: int,
        tool: Optional[str] = None,
        tool_inventory: Optional[dict] = None,
    ) -> BackendResult:
        ...
