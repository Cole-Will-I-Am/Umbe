"""The Verifier — spec §1.3.

Independent verification of Executor output. The Verifier is a separate
inference path — it does not share chain-of-thought with the Executor.

Signals instrumented (§1.3):
    confidence       — token entropy + trajectory disagreement across
                       N sampled completions, calibrated to probability.
    evidence_support — retrieval hit density for claims in output.
    coherence        — contradiction classifier flags + severity.
    friction         — count of replanning events + tool errors + dead ends.
    novelty          — embedding distance to nearest episodic cluster.

Hard invariants:
    * Verifier does not modify Executor output. It scores.
    * If the score falls below threshold, control returns to the Planner
      for a different approach (runtime-level decision).
    * The Verifier is gated by the Scheduler. If verification gate is
      closed, verify() returns a skip result.
    * Verifier cannot be bypassed for critical-stakes tasks (§5 Honesty).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from .backends.base import InferenceBackend
from .backends.stub import StubBackend
from .memory import EpisodicMemory
from .memory.embedding import cosine, embed
from .types import ExecutionTrace, Stakes, Task


# ---------------------------------------------------------------------------
# Coherence classifier — pluggable
# ---------------------------------------------------------------------------
@runtime_checkable
class CoherenceClassifier(Protocol):
    """Classify self-contradiction inside a text output.

    Returns ``(severity, contradictions)`` where severity ∈ [0, 1]
    (0 = clean, 1 = severe) and ``contradictions`` is a list of the
    evidence fragments that triggered the verdict.

    Implementations live outside the Verifier so the Verifier's scoring
    interface stays stable while the classifier is upgraded from
    marker-phrase heuristics to a real NLI model.
    """

    def classify(self, output: str) -> tuple[float, list[str]]:  # pragma: no cover
        ...


class MarkerCoherenceClassifier:
    """Zero-dep default: flags a small set of contradiction markers.

    This is deliberately conservative — it preserves the exact behavior
    of the original Verifier. Upgrade by constructing the Verifier with
    an ``LLMCoherenceClassifier`` or your own learned implementation.
    """

    DEFAULT_MARKERS: tuple[str, ...] = (
        "however contradicts",
        "actually wrong",
        "no it is",
        "but earlier said",
    )

    def __init__(self, markers: Optional[tuple[str, ...]] = None) -> None:
        self.markers = markers or self.DEFAULT_MARKERS

    def classify(self, output: str) -> tuple[float, list[str]]:
        if not output:
            return 0.0, []
        lowered = output.lower()
        hits = [m for m in self.markers if m in lowered]
        severity = min(1.0, 0.25 * len(hits))
        return severity, hits


class LLMCoherenceClassifier:
    """Backend-driven contradiction detector.

    Asks an inference backend (a *verifier* backend, disjoint from the
    Executor's, per §1.3) whether the text contradicts itself. Expects a
    structured YES/NO answer followed by optional contradiction excerpts.

    The backend handle is reused across calls so the cost is one extra
    inference per verification. The Scheduler's verification budget
    covers it.
    """

    PROMPT = (
        "You are a strict coherence checker. The following text is the "
        "output of an AI reasoning step. Determine whether it contains "
        "internal contradictions — claims that cannot simultaneously "
        "be true.\n\n"
        "Respond in this exact format:\n"
        "VERDICT: YES or NO\n"
        "SEVERITY: 0.0 to 1.0\n"
        "CONTRADICTIONS: semicolon-separated fragments, or NONE\n\n"
        "Text:\n{text}\n"
    )

    def __init__(
        self,
        backend: InferenceBackend,
        max_tokens: int = 256,
    ) -> None:
        self.backend = backend
        self.max_tokens = max_tokens

    def classify(self, output: str) -> tuple[float, list[str]]:
        if not output:
            return 0.0, []
        prompt = self.PROMPT.format(text=output)
        try:
            result = self.backend.run(prompt=prompt, max_tokens=self.max_tokens)
        except Exception:
            # Fall back to "no verdict" — let the marker classifier or
            # score combiner decide. We never raise out of the Verifier.
            return 0.0, []
        return _parse_coherence_response(result.output or "")


def _parse_coherence_response(text: str) -> tuple[float, list[str]]:
    """Parse the strict-format response from LLMCoherenceClassifier.

    Tolerant of ordering and surrounding prose — anchored on the
    VERDICT/SEVERITY/CONTRADICTIONS labels.
    """
    verdict_match = re.search(r"VERDICT:\s*(YES|NO)", text, re.IGNORECASE)
    severity_match = re.search(r"SEVERITY:\s*([0-9]*\.?[0-9]+)", text)
    contr_match = re.search(r"CONTRADICTIONS:\s*(.+)", text, re.IGNORECASE)

    verdict = (verdict_match.group(1).upper() if verdict_match else "NO")
    try:
        severity = float(severity_match.group(1)) if severity_match else 0.0
    except ValueError:
        severity = 0.0
    severity = max(0.0, min(1.0, severity))
    if verdict == "NO":
        severity = 0.0

    contradictions: list[str] = []
    if contr_match:
        raw = contr_match.group(1).strip()
        if raw and raw.upper() != "NONE":
            contradictions = [c.strip() for c in raw.split(";") if c.strip()]

    if verdict == "YES" and severity == 0.0:
        # Guarantee a non-zero severity when the model says there's a
        # contradiction — lets the score combiner actually penalise it.
        severity = 0.5
    return severity, contradictions


@dataclass
class VerificationResult:
    score: float  # 0..1 — overall pass quality
    confidence: float
    evidence_support: float
    coherence_severity: float  # 0 clean, 1 severe
    friction: int
    novelty: float
    contradictions: list[str] = field(default_factory=list)
    unsupported_assertions: list[str] = field(default_factory=list)
    passed: bool = True
    rationale: str = ""
    skipped: bool = False

    @classmethod
    def skip(cls) -> "VerificationResult":
        return cls(
            score=1.0,
            confidence=0.0,
            evidence_support=0.0,
            coherence_severity=0.0,
            friction=0,
            novelty=0.0,
            passed=True,
            rationale="gate_closed",
            skipped=True,
        )


@dataclass
class VerifierConfig:
    pass_threshold: float = 0.6
    # Contradiction keywords the default marker classifier flags.
    # Retained as a config knob so callers can tune the marker set
    # without subclassing MarkerCoherenceClassifier.
    contradiction_markers: tuple[str, ...] = MarkerCoherenceClassifier.DEFAULT_MARKERS
    claim_markers: tuple[str, ...] = ("is", "are", "was", "were", "will", "has")
    critical_always_verify: bool = True
    # Pluggable coherence classifier (§1.3). If None, the Verifier uses
    # a MarkerCoherenceClassifier seeded from contradiction_markers.
    coherence_classifier: Optional[CoherenceClassifier] = None


class Verifier:
    """Independent verifier. Takes a separate backend handle so its
    chain-of-thought is physically disjoint from the Executor's.
    """

    def __init__(
        self,
        backend: Optional[InferenceBackend] = None,
        config: Optional[VerifierConfig] = None,
        episodic: Optional[EpisodicMemory] = None,
    ) -> None:
        # NOTE: default to a *fresh* StubBackend so that the Verifier does
        # not accidentally share state with the Executor's backend. Runtime
        # wiring passes an explicit, distinct backend in real deployments.
        self.backend: InferenceBackend = backend or StubBackend()
        self.config = config or VerifierConfig()
        self.episodic = episodic
        # Resolve the coherence classifier. Explicit wins; otherwise
        # build a marker classifier from the config's marker tuple so
        # existing callers that only tweak contradiction_markers keep
        # working.
        self.coherence_classifier: CoherenceClassifier = (
            self.config.coherence_classifier
            or MarkerCoherenceClassifier(self.config.contradiction_markers)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def verify(
        self,
        task: Task,
        trace: ExecutionTrace,
        gate_open: bool,
    ) -> VerificationResult:
        # Critical stakes always verify — spec §5 Honesty invariant.
        if not gate_open and not (
            self.config.critical_always_verify and task.stakes == Stakes.CRITICAL
        ):
            return VerificationResult.skip()

        final_output = _final_output(trace)
        confidence = self._confidence(trace)
        evidence_support, unsupported = self._evidence_support(final_output, trace)
        coherence_severity, contradictions = self._coherence(final_output)
        friction = self._friction(trace)
        novelty = self._novelty(task)

        score = _combine_signals(
            confidence=confidence,
            evidence_support=evidence_support,
            coherence_severity=coherence_severity,
            friction=friction,
        )
        passed = score >= self.config.pass_threshold
        rationale = (
            f"conf={confidence:.2f} ev={evidence_support:.2f} "
            f"coh={1.0 - coherence_severity:.2f} fric={friction} "
            f"novelty={novelty:.2f}"
        )
        return VerificationResult(
            score=score,
            confidence=confidence,
            evidence_support=evidence_support,
            coherence_severity=coherence_severity,
            friction=friction,
            novelty=novelty,
            contradictions=contradictions,
            unsupported_assertions=unsupported,
            passed=passed,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------
    def _confidence(self, trace: ExecutionTrace) -> float:
        """Calibrated confidence from per-step entropy + trajectory
        disagreement. We approximate the "N sampled completions
        disagreement" part by calling the backend a second time on the
        final output's context and measuring embedding distance."""
        if not trace.step_results:
            return 0.0
        mean_entropy = sum(s.entropy for s in trace.step_results) / len(
            trace.step_results
        )
        # Map entropy ∈ [0, 1] → confidence ∈ [0, 1] (lower entropy = higher
        # confidence).
        entropy_confidence = max(0.0, min(1.0, 1.0 - mean_entropy))

        # Trajectory disagreement: re-sample the final step's prompt and
        # compare embeddings.
        last = trace.step_results[-1]
        if not last.output:
            return entropy_confidence
        try:
            resample = self.backend.run(
                prompt=last.output,
                max_tokens=min(128, last.tokens_used or 128),
            )
            agreement = cosine(embed(last.output), embed(resample.output))
        except Exception:
            agreement = 0.5
        # Average the two channels.
        return (entropy_confidence + max(0.0, agreement)) / 2.0

    def _evidence_support(
        self, output: str, trace: ExecutionTrace
    ) -> tuple[float, list[str]]:
        """Retrieval hit density: of all sentence-level claims in the
        output, what fraction appear to be supported by evidence surfaced
        during the run (tool outputs + working memory content)."""
        if not output:
            return 0.0, []
        claims = _extract_claims(output, self.config.claim_markers)
        if not claims:
            return 1.0, []

        corpus_tokens = set()
        for sr in trace.step_results:
            for tc in sr.tool_calls:
                corpus_tokens.update(_tokens(str(tc)))
            corpus_tokens.update(_tokens(sr.output or ""))

        supported = 0
        unsupported: list[str] = []
        for c in claims:
            c_tokens = _tokens(c)
            if not c_tokens:
                continue
            overlap = len(c_tokens & corpus_tokens) / len(c_tokens)
            if overlap >= 0.5:
                supported += 1
            else:
                unsupported.append(c)
        return supported / len(claims), unsupported

    def _coherence(self, output: str) -> tuple[float, list[str]]:
        return self.coherence_classifier.classify(output)

    def _friction(self, trace: ExecutionTrace) -> int:
        tool_errors = sum(
            1
            for sr in trace.step_results
            for tc in sr.tool_calls
            if not tc.get("success", True)
        )
        dead_ends = sum(1 for sr in trace.step_results if not sr.success)
        return trace.replanning_events + tool_errors + dead_ends

    def _novelty(self, task: Task) -> float:
        if not self.episodic or len(self.episodic) == 0:
            return 1.0  # No history means everything is novel.
        q = embed(task.input)
        best = 0.0
        for ep in self.episodic.all():
            if not ep.embedding:
                continue
            sim = cosine(q, ep.embedding)
            if sim > best:
                best = sim
        return max(0.0, 1.0 - best)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _final_output(trace: ExecutionTrace) -> str:
    if not trace.step_results:
        return ""
    return trace.step_results[-1].output or ""


def _extract_claims(text: str, markers: tuple[str, ...]) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[str] = []
    for s in sentences:
        if not s:
            continue
        lowered = s.lower()
        if any(f" {m} " in f" {lowered} " for m in markers) or s.endswith("."):
            out.append(s)
    return out


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _combine_signals(
    confidence: float,
    evidence_support: float,
    coherence_severity: float,
    friction: int,
) -> float:
    coherence = 1.0 - coherence_severity
    friction_penalty = min(0.5, 0.1 * friction)
    score = (
        0.40 * confidence
        + 0.35 * evidence_support
        + 0.25 * coherence
        - friction_penalty
    )
    return max(0.0, min(1.0, score))
