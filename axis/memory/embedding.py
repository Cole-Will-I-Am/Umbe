"""Embedding providers for similarity search.

This module defines a pluggable ``EmbeddingProvider`` protocol and a
default zero-dependency implementation (hashed bag-of-words) so the core
path keeps its "no external dependencies" guarantee. Optional providers
— notably ``SentenceTransformerProvider`` — are guarded behind a lazy
import and are a drop-in upgrade when a real embedding model is
installed.

All AXIS components that previously called ``embed(text)`` continue to
work unchanged; they now resolve through the active default provider.
Swap providers process-wide with ``set_default_provider(...)`` or pass a
specific provider via the ``provider=`` keyword.

The default provider's output is byte-for-byte identical to the previous
``embed`` implementation so existing tests and serialized episodes stay
consistent.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Optional, Protocol, runtime_checkable

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_DIM = 64


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal protocol: turn a string into a fixed-length vector."""

    dim: int

    def encode(self, text: str) -> list[float]:  # pragma: no cover - protocol
        ...


class HashedBagOfWordsProvider:
    """Deterministic zero-dependency embedding.

    This is the fallback / default provider. It is NOT a language-model
    embedding — it is a hashed bag-of-words vector with random sign
    projection and L2 normalisation. Useful for:

        * tests that must be reproducible without network access
        * ensuring the core AXIS path has no external dependencies
        * bootstrap before a real embedding backend is configured
    """

    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim

    def encode(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _WORD_RE.findall(text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            h = int.from_bytes(
                hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big"
            )
            idx = h % self.dim
            sign = 1.0 if (h >> 32) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]


class SentenceTransformerProvider:
    """Real sentence-transformers backed embedding.

    Optional — requires ``pip install sentence-transformers``. Loads the
    model lazily on first ``encode`` so importing this module never
    triggers a download.

    Example:

        from axis.memory.embedding import (
            SentenceTransformerProvider,
            set_default_provider,
        )
        set_default_provider(SentenceTransformerProvider("all-MiniLM-L6-v2"))
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self.dim = 0  # populated on first encode

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "SentenceTransformerProvider requires the "
                "'sentence-transformers' package. Install with "
                "`pip install sentence-transformers`."
            ) from e
        self._model = SentenceTransformer(self.model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> list[float]:  # pragma: no cover - optional dep
        self._ensure_model()
        assert self._model is not None
        vec = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vec]


# ---------------------------------------------------------------------------
# Module-level default provider
# ---------------------------------------------------------------------------
_default_provider: EmbeddingProvider = HashedBagOfWordsProvider(_DIM)


def set_default_provider(provider: EmbeddingProvider) -> None:
    """Swap the process-wide default embedding provider.

    Affects every subsequent call to ``embed()`` that does not pass an
    explicit ``provider=`` argument. Previously encoded vectors stored in
    memory are NOT re-embedded — cosine similarity across providers is
    undefined, so swap deliberately (e.g. at startup).
    """
    global _default_provider
    _default_provider = provider


def get_default_provider() -> EmbeddingProvider:
    return _default_provider


def embed(
    text: str,
    dim: Optional[int] = None,
    provider: Optional[EmbeddingProvider] = None,
) -> list[float]:
    """Return an embedding vector for ``text``.

    ``provider`` defaults to the module's default provider. ``dim`` is
    accepted for backwards compatibility with the old hashed-only API; it
    only affects the hashed provider and is ignored by learned
    providers.
    """
    if provider is None:
        provider = _default_provider
    if dim is not None and isinstance(provider, HashedBagOfWordsProvider):
        # Respect the legacy override for the hashed provider so old
        # callers keep working.
        return HashedBagOfWordsProvider(dim).encode(text)
    return provider.encode(text)


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # Providers normalise on output; we still clamp defensively.
    return max(-1.0, min(1.0, dot))
