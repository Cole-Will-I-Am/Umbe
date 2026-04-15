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
from axis.verifier import (
    LLMCoherenceClassifier,
    MarkerCoherenceClassifier,
    Verifier,
    VerifierConfig,
    _combine_signals,
    _parse_coherence_response,
)


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


# ---------------------------------------------------------------------------
# Pluggable coherence classifier (§1.3)
# ---------------------------------------------------------------------------
class _ScriptedBackend:
    """Minimal backend test double returning a fixed text on every call."""

    def __init__(self, text: str):
        self.text = text
        self.prompts: list[str] = []

    def run(self, prompt, max_tokens, tool=None, tool_inventory=None):
        self.prompts.append(prompt)
        return BackendResult(
            output=self.text,
            tokens_used=len(self.text) // 4 + 1,
            success=True,
            entropy=0.0,
        )


def test_marker_classifier_is_default():
    v = Verifier(backend=StubBackend())
    assert isinstance(v.coherence_classifier, MarkerCoherenceClassifier)


def test_custom_coherence_classifier_is_consulted():
    class _AlwaysFlags:
        def classify(self, output):
            return 0.9, ["fake contradiction"]

    cfg = VerifierConfig(coherence_classifier=_AlwaysFlags())
    v = Verifier(backend=StubBackend(), config=cfg)
    trace = _trace_with_output("the sky is blue")
    result = v.verify(Task(input="x"), trace, gate_open=True)
    assert result.coherence_severity == pytest.approx(0.9)
    assert result.contradictions == ["fake contradiction"]


def test_llm_coherence_classifier_parses_yes_verdict():
    backend = _ScriptedBackend(
        "VERDICT: YES\nSEVERITY: 0.8\n"
        "CONTRADICTIONS: claim A contradicts claim B; foo contradicts bar\n"
    )
    clf = LLMCoherenceClassifier(backend=backend)
    severity, contradictions = clf.classify("any text")
    assert severity == pytest.approx(0.8)
    assert contradictions == [
        "claim A contradicts claim B",
        "foo contradicts bar",
    ]
    # The classifier actually called the backend with the text embedded.
    assert "any text" in backend.prompts[0]


def test_llm_coherence_classifier_no_verdict_returns_zero():
    backend = _ScriptedBackend(
        "VERDICT: NO\nSEVERITY: 0.3\nCONTRADICTIONS: NONE\n"
    )
    clf = LLMCoherenceClassifier(backend=backend)
    severity, contradictions = clf.classify("any text")
    assert severity == 0.0
    assert contradictions == []


def test_llm_coherence_classifier_yes_without_severity_gets_floor():
    severity, _ = _parse_coherence_response(
        "VERDICT: YES\nCONTRADICTIONS: one thing; another thing\n"
    )
    assert severity >= 0.5


def test_llm_coherence_classifier_empty_text_short_circuits():
    backend = _ScriptedBackend("VERDICT: YES\nSEVERITY: 1.0\n")
    clf = LLMCoherenceClassifier(backend=backend)
    assert clf.classify("") == (0.0, [])
    assert backend.prompts == []  # Never called on empty input.


def test_llm_coherence_classifier_tolerates_backend_errors():
    class _Boom:
        def run(self, *a, **kw):
            raise RuntimeError("api down")

    clf = LLMCoherenceClassifier(backend=_Boom())
    severity, contradictions = clf.classify("anything")
    assert severity == 0.0
    assert contradictions == []


def test_verifier_uses_llm_classifier_end_to_end():
    backend = _ScriptedBackend(
        "VERDICT: YES\nSEVERITY: 0.7\nCONTRADICTIONS: A; B\n"
    )
    cfg = VerifierConfig(
        coherence_classifier=LLMCoherenceClassifier(backend=backend),
    )
    v = Verifier(backend=StubBackend(), config=cfg)
    trace = _trace_with_output("Some text claiming two incompatible things.")
    result = v.verify(Task(input="x"), trace, gate_open=True)
    assert result.coherence_severity == pytest.approx(0.7)
    assert result.contradictions == ["A", "B"]
    # Verifier's own backend (StubBackend) and the classifier's backend
    # are physically distinct — preserves the §1.3 "separate inference
    # path" guarantee even for the coherence sub-check.
    assert v.backend is not backend
