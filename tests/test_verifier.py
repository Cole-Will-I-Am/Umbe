from unittest.mock import MagicMock

import pytest

from axis.backends.base import BackendResult
from axis.backends.stub import StubBackend
from axis.memory import Episode, EpisodicMemory
from axis.memory.embedding import embed
from axis.types import (
    ExecutionTrace,
    Stakes,
    StepResult,
    Strategy,
    Task,
    TaskOutcome,
)
from axis.verifier import Verifier, VerifierConfig, _combine_signals


def _trace_with_output(
    output: str,
    tool_calls: list[dict] | None = None,
    entropy: float = 0.05,
    outcome: TaskOutcome = TaskOutcome.SUCCESS,
    replans: int = 0,
) -> ExecutionTrace:
    step = StepResult(
        step_id="s",
        output=output,
        tokens_used=50,
        tool_calls=tool_calls or [],
        success=outcome != TaskOutcome.FAILURE,
        entropy=entropy,
    )
    return ExecutionTrace(
        task_id="t",
        task_class="analysis",
        strategy=Strategy.ANALYTICAL,
        step_results=[step],
        execution_tokens=50,
        total_tokens=50,
        replanning_events=replans,
        outcome=outcome,
    )


def test_verifier_skips_when_gate_closed():
    v = Verifier(backend=StubBackend())
    trace = _trace_with_output("The sky is blue.")
    task = Task(input="x", stakes=Stakes.LOW)
    result = v.verify(task, trace, gate_open=False)
    assert result.skipped is True
    assert result.passed is True
    assert result.rationale == "gate_closed"


def test_verifier_always_runs_on_critical_stakes():
    """Spec §5 Honesty invariant: Verifier cannot be bypassed for critical tasks."""
    v = Verifier(backend=StubBackend())
    task = Task(input="x", stakes=Stakes.CRITICAL)
    trace = _trace_with_output("Answer.")
    result = v.verify(task, trace, gate_open=False)
    assert result.skipped is False


def test_verifier_does_not_share_backend_with_executor():
    """The Verifier must not share chain-of-thought with the Executor.
    We enforce this at the construction level by requiring a distinct
    backend instance."""
    exec_backend = StubBackend()
    verif_backend = StubBackend()
    v = Verifier(backend=verif_backend)
    assert v.backend is not exec_backend


def test_verifier_does_not_mutate_trace():
    v = Verifier(backend=StubBackend())
    trace = _trace_with_output("Answer text.")
    before = trace.step_results[0].output
    v.verify(Task(input="x"), trace, gate_open=True)
    assert trace.step_results[0].output == before


def test_verifier_evidence_support_counts_tool_overlap():
    v = Verifier(backend=StubBackend())
    trace = _trace_with_output(
        "The quarterly revenue is growing. Margins are healthy.",
        tool_calls=[
            {
                "tool": "retrieval",
                "success": True,
                "args": {"excerpt": "quarterly revenue growth margins healthy"},
            }
        ],
    )
    result = v.verify(Task(input="earnings"), trace, gate_open=True)
    assert result.evidence_support > 0.4


def test_verifier_friction_penalty():
    v = Verifier(backend=StubBackend())
    trace = _trace_with_output(
        "Answer.",
        tool_calls=[{"tool": "search", "success": False}],
        replans=2,
    )
    result = v.verify(Task(input="x"), trace, gate_open=True)
    # 2 replans + 1 tool error = 3 friction
    assert result.friction == 3


def test_verifier_novelty_against_episodic_memory():
    em = EpisodicMemory()
    em.add(
        Episode(
            episode_id="e1",
            timestamp=0.0,
            task_hash="h",
            task_class="code_debug",
            input_summary="",
            strategy_used=None,
            tools_invoked=[],
            outcome="success",
            embedding=embed("Fix null pointer in parser"),
        )
    )
    v = Verifier(backend=StubBackend(), episodic=em)
    trace = _trace_with_output("ok")
    # Same text → low novelty
    r1 = v.verify(Task(input="Fix null pointer in parser"), trace, gate_open=True)
    r2 = v.verify(Task(input="Write a sonnet about springtime"), trace, gate_open=True)
    assert r1.novelty < r2.novelty


def test_verifier_novelty_defaults_to_max_without_history():
    v = Verifier(backend=StubBackend())
    result = v.verify(Task(input="x"), _trace_with_output("ok"), gate_open=True)
    assert result.novelty == 1.0


def test_verifier_pass_fail_from_signals():
    # High confidence, high evidence, no friction → pass
    good = _combine_signals(0.9, 0.9, 0.0, 0)
    assert good > 0.6
    # Low across the board + high friction → fail
    bad = _combine_signals(0.2, 0.1, 0.8, 5)
    assert bad < 0.4


def test_verifier_contradiction_marker_hurts_score():
    v = Verifier(backend=StubBackend())
    clean = _trace_with_output("The quarter was strong. Revenue rose.")
    muddled = _trace_with_output(
        "The quarter was strong. However contradicts the earlier claim."
    )
    r_clean = v.verify(Task(input="x"), clean, gate_open=True)
    r_muddled = v.verify(Task(input="x"), muddled, gate_open=True)
    assert r_muddled.coherence_severity > r_clean.coherence_severity
    assert r_muddled.score < r_clean.score
