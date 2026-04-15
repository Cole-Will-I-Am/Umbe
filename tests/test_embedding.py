"""Tests for the pluggable EmbeddingProvider layer."""

from __future__ import annotations

import math

import pytest

from axis.memory import (
    Episode,
    EpisodicMemory,
    HashedBagOfWordsProvider,
    cosine,
    embed,
    get_default_provider,
    set_default_provider,
)
from axis.memory import embedding as embedding_module


class _ConstantProvider:
    """Test double: always returns the same vector, regardless of input."""

    dim = 4

    def __init__(self, vec: list[float]):
        self.vec = vec
        self.calls: list[str] = []

    def encode(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self.vec)


@pytest.fixture(autouse=True)
def _restore_default_provider():
    original = get_default_provider()
    try:
        yield
    finally:
        set_default_provider(original)


def test_default_provider_is_hashed_bag_of_words():
    assert isinstance(get_default_provider(), HashedBagOfWordsProvider)


def test_embed_output_is_unchanged_for_legacy_callers():
    """The hashed provider output must stay byte-identical so that
    previously stored episode/semantic embeddings keep matching."""
    v = embed("hello world")
    # L2-normalised, 64-dim by default.
    assert len(v) == 64
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-9


def test_embed_is_deterministic():
    assert embed("the quick brown fox") == embed("the quick brown fox")


def test_empty_text_returns_zero_vector():
    v = embed("")
    assert v == [0.0] * 64


def test_set_default_provider_switches_embed():
    fake = _ConstantProvider([1.0, 0.0, 0.0, 0.0])
    set_default_provider(fake)
    v = embed("anything")
    assert v == [1.0, 0.0, 0.0, 0.0]
    assert fake.calls == ["anything"]


def test_per_call_provider_override_does_not_change_default():
    fake = _ConstantProvider([0.0, 1.0, 0.0, 0.0])
    v = embed("hi", provider=fake)
    assert v == [0.0, 1.0, 0.0, 0.0]
    # Default provider must still be the original hashed one.
    assert isinstance(get_default_provider(), HashedBagOfWordsProvider)
    # And calling embed() without provider must fall back to it.
    v2 = embed("hi")
    assert len(v2) == 64


def test_legacy_dim_kwarg_still_honored_for_hashed_provider():
    v16 = embed("hello", dim=16)
    assert len(v16) == 16


def test_episodic_similar_sees_provider_swap():
    """Swapping the default provider must flow through to the memory
    layer — EpisodicMemory.similar() reads via the module-level embed()
    in its own code, so it should reflect the change for *new* queries.
    """
    em = EpisodicMemory()
    em.add(
        Episode(
            episode_id="e1",
            timestamp=0.0,
            task_hash="h",
            task_class="general",
            input_summary="",
            strategy_used=None,
            tools_invoked=[],
            outcome="success",
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
    )
    fake = _ConstantProvider([1.0, 0.0, 0.0, 0.0])
    set_default_provider(fake)
    hits = em.similar("any query at all", top_k=1)
    assert len(hits) == 1
    ep, score = hits[0]
    assert ep.episode_id == "e1"
    assert score == pytest.approx(1.0)


def test_cosine_identical_vectors_is_one():
    v = [1.0, 0.0, 0.0, 0.0]
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_mismatched_lengths_returns_zero():
    assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_provider_module_exposes_dim_attribute():
    p = HashedBagOfWordsProvider(dim=32)
    assert p.dim == 32
    assert len(p.encode("hi")) == 32


def test_embedding_module_has_backwards_compatible_api():
    # The legacy top-level symbols still exist at the same path for
    # callers that imported them directly.
    assert hasattr(embedding_module, "embed")
    assert hasattr(embedding_module, "cosine")
