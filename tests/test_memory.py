import time

import pytest

from axis.backends.stub import StubBackend
from axis.executor import Executor
from axis.memory import (
    Episode,
    EpisodicMemory,
    ForgettingEngine,
    MemoryManager,
    ProceduralMemory,
    SemanticMemory,
)
from axis.memory.embedding import cosine, embed
from axis.memory.episodic import episode_from_trace
from axis.memory.forgetting import ForgettingPolicy
from axis.runtime import AxisRuntime
from axis.types import (
    ExecutionTrace,
    Stakes,
    StepResult,
    Strategy,
    Task,
    TaskOutcome,
)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def test_embedding_deterministic_and_normalised():
    a = embed("Find the bug in module X")
    b = embed("Find the bug in module X")
    c = embed("Write a poem about clouds")
    assert a == b
    assert cosine(a, b) == pytest.approx(1.0)
    assert cosine(a, c) < 0.9


def test_embedding_empty_text_is_zero_vector():
    v = embed("")
    assert all(x == 0.0 for x in v)


# ---------------------------------------------------------------------------
# Episodic memory
# ---------------------------------------------------------------------------
_make_episode_counter = {"n": 0}


def _make_episode(
    task_class="code_debug",
    outcome="success",
    text="Fix the null pointer at line 42",
):
    _make_episode_counter["n"] += 1
    return Episode(
        episode_id=f"ep-{_make_episode_counter['n']}-{outcome}",
        timestamp=time.time(),
        task_hash="h",
        task_class=task_class,
        input_summary=text,
        strategy_used="analytical",
        tools_invoked=["code_search"],
        outcome=outcome,
        outcome_detail=None if outcome == "success" else "bad_fix",
        embedding=embed(text),
    )


def test_episodic_add_and_retrieve():
    em = EpisodicMemory()
    ep = _make_episode()
    em.add(ep)
    assert len(em) == 1
    assert em.by_class("code_debug") == [ep]
    assert em.by_outcome("success") == [ep]


def test_episodic_similar_ranks_related_first():
    em = EpisodicMemory()
    em.add(_make_episode(text="Fix null pointer exception in parser"))
    em.add(_make_episode(text="Write a haiku about autumn leaves"))
    results = em.similar("null pointer parser bug")
    assert results[0][0].input_summary.startswith("Fix null")


def test_episodic_cluster_failures_groups_by_error():
    em = EpisodicMemory()
    for i in range(3):
        em.add(
            Episode(
                episode_id=f"f{i}",
                timestamp=time.time(),
                task_hash="h",
                task_class="code_debug",
                input_summary="x",
                strategy_used="analytical",
                tools_invoked=[],
                outcome="failure",
                outcome_detail="timeout",
            )
        )
    em.add(
        Episode(
            episode_id="other",
            timestamp=time.time(),
            task_hash="h",
            task_class="code_debug",
            input_summary="x",
            strategy_used="analytical",
            tools_invoked=[],
            outcome="failure",
            outcome_detail="oom",
        )
    )
    clusters = em.cluster_failures("code_debug")
    assert len(clusters["timeout"]) == 3
    assert len(clusters["oom"]) == 1


def test_episode_from_trace_copies_fields():
    trace = ExecutionTrace(
        task_id="t",
        task_class="analysis",
        strategy=Strategy.RETRIEVAL_FIRST,
        step_results=[
            StepResult(
                step_id="s",
                output="ok",
                tokens_used=100,
                tool_calls=[{"tool": "retrieval", "success": True}],
                success=True,
            )
        ],
        execution_tokens=100,
        total_tokens=100,
        outcome=TaskOutcome.SUCCESS,
    )
    trace.start_time = 0.0
    trace.end_time = 0.5
    task = Task(input="Summarise earnings call", task_class="analysis")
    ep = episode_from_trace(trace, task)
    assert ep.task_class == "analysis"
    assert ep.strategy_used == "retrieval_first"
    assert "retrieval" in ep.tools_invoked
    assert ep.resource_cost["tokens"] == 100
    assert ep.outcome == "success"


def test_episodic_postmortem_annotation():
    em = EpisodicMemory()
    ep = _make_episode()
    em.add(ep)
    em.annotate_postmortem(ep.episode_id, "retrieval returned stale dep info")
    assert em.all()[0].postmortem == "retrieval returned stale dep info"


def test_episodic_duplicate_id_rejected():
    em = EpisodicMemory()
    ep = _make_episode()
    em.add(ep)
    with pytest.raises(ValueError):
        em.add(ep)


# ---------------------------------------------------------------------------
# Semantic memory
# ---------------------------------------------------------------------------
def test_semantic_upsert_versioning_and_similarity():
    sm = SemanticMemory()
    ent = sm.upsert(
        type="distilled_lesson",
        content="retrieval_first wins for legal_reasoning",
        confidence=0.8,
        source_episodes=["e1"],
    )
    assert ent.version == 1

    ent2 = sm.upsert(
        type="distilled_lesson",
        content="retrieval_first wins for legal_reasoning",
        confidence=0.9,
        source_episodes=["e2"],
    )
    assert ent2.entity_id == ent.entity_id
    assert ent2.version == 2
    assert set(ent2.source_episodes) == {"e1", "e2"}
    assert ent2.confidence == 0.9

    # similarity search
    hits = sm.similar("legal_reasoning retrieval")
    assert hits and hits[0][0].entity_id == ent.entity_id


def test_semantic_contradiction_resolution_picks_winner_by_sources_and_confidence():
    sm = SemanticMemory()
    a = sm.upsert(
        type="fact",
        content="A",
        confidence=0.9,
        source_episodes=["e1"],
        entity_id="id-a",
    )
    b = sm.upsert(
        type="fact",
        content="B",
        confidence=0.7,
        source_episodes=["e1", "e2", "e3"],
        entity_id="id-b",
    )
    winner = sm.resolve_contradiction("id-a", "id-b")
    assert winner.entity_id == "id-b"
    assert sm.get("id-a") is None


# ---------------------------------------------------------------------------
# Procedural memory
# ---------------------------------------------------------------------------
def test_procedural_register_record_usage_and_promote():
    pm = ProceduralMemory()
    proc = pm.register(
        name="debug_python_with_tests",
        task_class="code_debug",
        steps=[{"action": "run_tests"}],
    )
    assert proc.status == "candidate"
    pm.record_usage(proc.procedure_id, success=True, cost={"tokens": 500})
    pm.record_usage(proc.procedure_id, success=True, cost={"tokens": 700})
    proc_reloaded = pm.get(proc.procedure_id)
    assert proc_reloaded.success_rate == 1.0
    assert proc_reloaded.avg_cost["tokens"] == 600.0

    pm.promote(proc.procedure_id)
    assert pm.best_for_class("code_debug") == proc_reloaded
    assert proc_reloaded.status == "validated"

    # Candidates excluded from for_class by default
    candidate = pm.register("new", "code_debug", steps=[])
    assert candidate not in pm.for_class("code_debug")
    assert candidate in pm.for_class("code_debug", include_candidates=True)


# ---------------------------------------------------------------------------
# Forgetting engine
# ---------------------------------------------------------------------------
def test_forgetting_budget_enforcement_evicts_lowest_value():
    policy = ForgettingPolicy(episodic_max=2)
    em = EpisodicMemory()
    old_fail = _make_episode(text="old fail", outcome="failure")
    old_fail.timestamp = time.time() - 1_000_000
    em.add(old_fail)
    em.add(_make_episode(text="recent success 1"))
    em.add(_make_episode(text="recent success 2"))

    fe = ForgettingEngine(policy)
    fe.run(em, SemanticMemory(), ProceduralMemory())
    ids = {ep.episode_id for ep in em.all()}
    assert old_fail.episode_id not in ids
    assert len(em) == 2


def test_forgetting_retires_weak_stale_procedures():
    policy = ForgettingPolicy(
        procedural_min_success_rate=0.5,
        procedural_idle_retire_s=0.0,
    )
    pm = ProceduralMemory()
    proc = pm.register("weak", "code_debug", steps=[])
    pm.record_usage(proc.procedure_id, success=False)
    # Force stale
    proc.last_used = time.time() - 10_000

    fe = ForgettingEngine(policy)
    fe.run(EpisodicMemory(), SemanticMemory(), pm)
    assert pm.get(proc.procedure_id).status == "retired"


def test_distill_similar_episodes_creates_semantic_lesson():
    em = EpisodicMemory()
    sm = SemanticMemory()
    for i in range(3):
        em.add(
            Episode(
                episode_id=f"s{i}",
                timestamp=time.time(),
                task_hash="h",
                task_class="analysis",
                input_summary="earnings summary",
                strategy_used="retrieval_first",
                tools_invoked=[],
                outcome="success",
                embedding=embed("earnings summary"),
            )
        )
    fe = ForgettingEngine()
    n = fe.distill_similar_episodes(em, sm, "analysis", min_cluster_size=3)
    assert n == 1
    assert len(sm) == 1
    assert sm.all()[0].type == "distilled_lesson"


# ---------------------------------------------------------------------------
# MemoryManager + runtime integration
# ---------------------------------------------------------------------------
def test_memory_manager_records_episode_from_runtime():
    mm = MemoryManager()
    rt = AxisRuntime(executor=Executor(backend=StubBackend()), memory=mm)
    rt.run(Task(input="Analyze this", task_class="analysis"))
    assert len(mm.episodic) == 1
    ep = mm.episodic.all()[0]
    assert ep.task_class == "analysis"
    assert ep.outcome == "success"


def test_memory_manager_procedural_index_feeds_planner():
    mm = MemoryManager()
    proc = mm.register_procedure(
        name="analysis_template",
        task_class="analysis",
        steps=[
            {"action": "retrieve_context", "tool": "retrieval", "strategy": "retrieval_first"},
            {"action": "summarise", "strategy": "analytical"},
        ],
    )
    mm.procedural.promote(proc.procedure_id)

    rt = AxisRuntime(executor=Executor(backend=StubBackend()), memory=mm)
    trace = rt.run(Task(input="x", task_class="analysis"))
    # After run, the planner should have pulled the procedure into its view
    assert "analysis" in rt.planner.procedural_memory
    assert trace.outcome == TaskOutcome.SUCCESS


def test_forget_cycle_runs_from_manager():
    mm = MemoryManager()
    # Stale episode
    ep = _make_episode(text="ancient")
    ep.timestamp = time.time() - 365 * 24 * 3600
    mm.episodic.add(ep)
    report = mm.forget_cycle()
    assert report.episodic_compressed >= 1
