"""Cheap deterministic embeddings for similarity search.

This is the "embedding" used by Episodic and Semantic memory for
similarity retrieval. It is NOT a language model embedding — it's a
hashed bag-of-words vector so tests are deterministic and the memory
layer has zero external dependencies. Replace with a real embedding
backend when one is wired in.
"""

from __future__ import annotations

import hashlib
import math
import re

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_DIM = 64


def embed(text: str, dim: int = _DIM) -> list[float]:
    """Return a normalised hashed bag-of-words vector for `text`."""
    vec = [0.0] * dim
    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        return vec
    for tok in tokens:
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
        idx = h % dim
        sign = 1.0 if (h >> 32) & 1 else -1.0
        vec[idx] += sign
    # L2 normalise
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # Vectors are already normalised by embed(), but guard anyway.
    return max(-1.0, min(1.0, dot))
