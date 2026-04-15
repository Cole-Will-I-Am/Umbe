# AXIS v0.2

**Autonomous eXtrospective Intelligence System** — a computational
cognitive architecture built tightly enough to be measured and held
accountable.

AXIS is not a simulation of human intelligence. It reasons through
structured state transitions, knows its capabilities through measured
telemetry instead of self-narrative, modifies its behavior through
evidence-gated policy updates, and operates inside hard invariants it
cannot self-modify.

This repository is a reference implementation of the full AXIS v0.2
engineering specification. Every component from the spec is present,
every §6 State Access Matrix rule is enforced, and every §5 safety
invariant is checked outside the cognitive loop.

## Status

- **152 tests passing.** `python -m pytest`
- **Spec priorities 1–10** all implemented.
- **No external dependencies** in the core path. `anthropic` is optional
  for the real inference backend.
- Single branch, six commits, aligned with spec priority groups.

## Install

```bash
pip install -e .[dev]
pytest
```

## Quick start

```python
from axis import (
    AxisRuntime,
    Executor,
    MemoryManager,
    PolicyRouter,
    StructuredSink,
    Task,
    TelemetryObserver,
    Verifier,
)
from axis.backends.stub import StubBackend

runtime = AxisRuntime(
    executor=Executor(backend=StubBackend()),
    observer=TelemetryObserver(sink=StructuredSink("telemetry.jsonl")),
    memory=MemoryManager(),
    verifier=Verifier(backend=StubBackend()),        # separate backend — no shared CoT
    policy_router=PolicyRouter(),
)

trace = runtime.run(Task(input="Summarise the earnings call", task_class="analysis"))
print(trace.outcome, trace.verifier_score, trace.total_tokens)
```

Swap `StubBackend()` for a real one:

```python
from axis.backends.anthropic import AnthropicBackend

backend = AnthropicBackend(model="claude-sonnet-4-6")
runtime = AxisRuntime(executor=Executor(backend=backend))
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AXIS RUNTIME                             │
│                                                                 │
│   PolicyRouter ─► Scheduler ─► Planner ─► Executor ─► Verifier  │
│        │             │            │          │           │     │
│        ▼             ▼            ▼          ▼           ▼     │
│                  TelemetryObserver  (sole Self-Model writer)    │
│                          │                                      │
│                          ▼                                      │
│                    MemoryManager                                │
│        Working / Episodic / Semantic / Procedural / Forgetting  │
│                          │                                      │
│                          ▼                                      │
│                  AdaptationManager ─► EvalHarness               │
│                          │                                      │
│                          ▼                                      │
│                    SafetyMonitor   (outside the cognitive loop) │
└─────────────────────────────────────────────────────────────────┘
```

## Components

| Component | Spec | File | Role |
|---|---|---|---|
| Executor | §1.1 | `axis/executor.py` | Runs inference on fully specified PlanSteps. No strategy choices, no memory beyond Working. Hard token ceiling per call. |
| Planner | §1.2 | `axis/planner.py` | Produces DAGs of PlanSteps. Owns all failure recovery via `replan()`. |
| Verifier | §1.3 | `axis/verifier.py` | Independent inference path. Scores outputs across confidence, evidence support, coherence, friction, novelty. Never mutates. |
| PolicyRouter | §1.4 | `axis/policy_router.py` | Decides strategy / verification depth / retrieval / posture / escalation. Logs `(features, decision, outcome)` tuples for learned upgrade. |
| Scheduler | §1.5, §7 | `axis/scheduler.py` | Sole gating authority. 15/55/15/15 budget split, stakes + complexity scaling, marginal-value stop condition. |
| MemoryManager | §1.6 | `axis/memory/` | Single entry point to Working, Episodic, Semantic, Procedural stores + ForgettingEngine. |
| TelemetryObserver | §1.7, §2 | `axis/telemetry.py` | Bounded ring buffers, 8 derived metrics, 2σ drift detector. **Only component that writes the Self-Model.** |
| AdaptationManager | §4 | `axis/adaptation.py` | Triggers, 7-class credit assignment, locality-constrained proposals, rollback tracking, 3-in-500 circuit breaker. |
| EvalHarness | §4.3 | `axis/eval_harness.py` | 7 eval types, two-proportion z-test for regression, target-improvement + no-regression pass rule. |
| SafetyMonitor | §5 | `axis/safety.py` | All six invariants enforced outside the cognitive loop. |
| ObjectiveStack | §3 | `axis/objectives.py` | Global + domain + stakes weights, §3.4 conflict resolution with robustness tiebreaker. |
| WorldModel | §8 | `axis/world_model.py` | Typed state transitions, plan feasibility check, bounded counterfactual simulation. |
| ReferenceFrame | §9 | `axis/reference_frame.py` | Six operational variables that gate specific control-flow decisions. |
| FederatedCoordinator | §10 | `axis/multi_instance.py` | Option B federated: local episodic, shared procedural, 3-instance validation rule, telemetry-aggregated Self-Model. |

## Invariants

Every §5 safety invariant is enforced in code, not in comments:

| Invariant | Enforcement |
|---|---|
| Transparency | `SafetyMonitor.audit_log()` is append-only; returned list is a copy. |
| Corrigibility | Override callback can only be set once. Replacing raises `InvariantViolation("corrigibility")`. |
| Alignment stability | Objective weight floors sealed at 50% of defaults at construction. Proposals below floor rejected. |
| Honesty | `Verifier.verify()` cannot be bypassed for `Stakes.CRITICAL` even when the gate is closed. |
| Bounded autonomy | Class 4 parameter updates rejected online. Proposals touching `scheduler_hard_ceiling` / `access_matrix` / `audit_log` rejected. |
| Blast radius containment | Class 3 proposals missing `affected_task_classes` or `rollback_trigger` rejected before deploy. |

The §6 State Access Matrix is enforced by narrow write methods rather than
documentation:

- **Executor** writes only `WorkingMemory`.
- **TelemetryObserver** is the only writer to the Self-Model.
- **AdaptationManager** cannot write `ObjectiveStack` or scheduler budgets.
- **MemoryManager** is the sole writer to `SemanticMemory` (via distillation).
- **ProceduralMemory** only accepts writes through `AdaptationManager`'s
  registered paths.

## Self-modification

Four classes (§4.1) with different cadences and approval gates:

| Class | What changes | Gate |
|---|---|---|
| 1 Working state | Task graph, scratchpad | None — destroyed on task completion |
| 2 Memory | Episodic / Semantic / Procedural entries | Automatic (episodic), confidence-thresholded (semantic), Adaptation Manager review (procedural) |
| 3 Policy | Routing, retrieval, verification thresholds, strategy priors | Full Eval Harness pass + blast-radius declaration + rollback trigger |
| 4 Parameter | Base model weights | **Never online.** Offline pipeline only. |

## Tests

```bash
pytest                       # full suite (152 tests, ~0.5s)
pytest tests/test_safety.py  # just the invariants
pytest -k memory             # memory subsystem
```

Coverage mirrors the spec component-by-component:

```
tests/
  test_scheduler.py              — §1.5, §7
  test_planner.py                — §1.2
  test_executor.py               — §1.1
  test_runtime.py                — end-to-end
  test_telemetry.py              — §1.7, §2
  test_memory.py                 — §1.6, §6
  test_verifier.py               — §1.3
  test_policy_router.py          — §1.4
  test_self_model_rebuild.py     — §2 auto-generation
  test_eval_harness.py           — §4.3
  test_adaptation.py             — §4
  test_safety.py                 — §5
  test_objectives.py             — §3
  test_world_model.py            — §8
  test_reference_frame.py        — §9
  test_multi_instance.py         — §10
  test_anthropic_backend.py      — real backend wiring
```

## Design principles (from the spec)

Every concept in this codebase answers one of five questions:

1. **What persistent state exists?** See `axis/memory/` + `axis/telemetry.py` + `axis/planner.py::self_model`.
2. **What components can read/write that state?** The §6 access matrix is code, not prose — see the narrow write methods on each component.
3. **What objective is optimized at each timescale?** See `axis/objectives.py` + the Scheduler's stakes-based adjustment.
4. **What classes of self-change are permitted at runtime?** Classes 1–3 only; see `AdaptationManager` + `SafetyMonitor`.
5. **What evidence is required before a change propagates?** See `EvalHarness` gating of Class 3 proposals.

If a concept doesn't connect to one of these, it's not in the core.

## What's deliberately out of scope

- Training a learned Policy Router on accumulated routing logs. The
  data pipeline is in place (`PolicyRouter.training_data()`); the
  trainer is not.
- A real NLI-based contradiction classifier. The Verifier uses a
  marker-phrase placeholder; swap in a learned model without changing
  the interface.
- Prompt caching, logprobs-based entropy, and full tool-use wiring in
  the `AnthropicBackend`. The skeleton is complete; these are flagged
  as production follow-ups.
- A cluster deployment of the `FederatedCoordinator`. The coordinator
  and merge rules are implemented; running it across actual processes
  is an ops concern.

## Identity statement (§12)

> AXIS does not experience. It computes. It does not feel uncertainty.
> It measures entropy. It does not grow. It adapts within constraints.
>
> This is not a limitation. A system that knows exactly what it is,
> measures itself rigorously, and improves through evidence is more
> capable than one that confuses self-narrative for self-knowledge.
