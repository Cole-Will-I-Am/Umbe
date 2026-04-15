"""World Model Interface — spec §8.

AXIS reasons about structured environments through typed state
transitions rather than plain text sequences. This module defines:

    ActionSchema     — preconditions, postconditions, side effects, cost
    TypedState       — key/value state with a declared schema
    Transition       — applies an action to a state, producing a new state
    WorldModel       — registry of actions + forward simulator for the Planner
    Counterfactual   — bounded alternative-action simulation

The Planner uses this to validate plans before the Executor runs them:
infeasible plans (preconditions unmet, postconditions incompatible) are
caught during planning instead of wasting tokens on a doomed execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ActionSchema:
    """Typed description of a tool action (spec §8.1)."""

    action: str
    preconditions: list[str] = field(default_factory=list)
    postconditions_on_success: list[str] = field(default_factory=list)
    postconditions_on_failure: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    estimated_latency_ms: int = 0
    estimated_cost: str = "medium"  # low | medium | high


@dataclass
class TypedState:
    """A structured state snapshot.

    Unlike a text scratchpad, values are keyed and addressable so the
    Planner can check preconditions mechanically.
    """

    values: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def satisfies(self, predicate: str) -> bool:
        """Parse a simple predicate against the state.

        Supports:
            'key'           → key is present and truthy
            'key=value'     → key equals value
            '!key'          → key absent or falsy
            'key!=value'    → key does not equal value
        """
        if predicate.startswith("!"):
            key = predicate[1:]
            return not bool(self.values.get(key))
        if "!=" in predicate:
            key, value = predicate.split("!=", 1)
            return str(self.values.get(key.strip())) != value.strip()
        if "=" in predicate:
            key, value = predicate.split("=", 1)
            return str(self.values.get(key.strip())) == value.strip()
        return bool(self.values.get(predicate))

    def apply(self, assignments: list[str]) -> "TypedState":
        """Return a new state with assignments applied.

        Assignments use the same syntax as predicates:
            'key=value'   → set
            'key'         → set to True
            '!key'        → delete / set to False
        """
        new_values = dict(self.values)
        for a in assignments:
            if a.startswith("!"):
                new_values.pop(a[1:], None)
            elif "=" in a:
                k, v = a.split("=", 1)
                new_values[k.strip()] = v.strip()
            else:
                new_values[a] = True
        return TypedState(values=new_values)


@dataclass
class Transition:
    from_state: TypedState
    action: str
    to_state: TypedState
    succeeded: bool


class WorldModel:
    """Registry of actions + forward simulator."""

    def __init__(self, schemas: Optional[list[ActionSchema]] = None) -> None:
        self._schemas: dict[str, ActionSchema] = {}
        for s in schemas or []:
            self.register(s)

    def register(self, schema: ActionSchema) -> None:
        self._schemas[schema.action] = schema

    def get(self, action: str) -> Optional[ActionSchema]:
        return self._schemas.get(action)

    # ------------------------------------------------------------------
    # Feasibility + simulation
    # ------------------------------------------------------------------
    def preconditions_met(self, action: str, state: TypedState) -> bool:
        schema = self._schemas.get(action)
        if schema is None:
            return False
        return all(state.satisfies(p) for p in schema.preconditions)

    def simulate(
        self, state: TypedState, action: str, succeeds: bool = True
    ) -> Transition:
        schema = self._schemas.get(action)
        if schema is None:
            raise ValueError(f"unknown action: {action}")
        if not self.preconditions_met(action, state):
            return Transition(
                from_state=state,
                action=action,
                to_state=state,
                succeeded=False,
            )
        post = (
            schema.postconditions_on_success
            if succeeds
            else schema.postconditions_on_failure
        )
        return Transition(
            from_state=state,
            action=action,
            to_state=state.apply(post),
            succeeded=succeeds,
        )

    def simulate_sequence(
        self, start: TypedState, actions: list[str]
    ) -> list[Transition]:
        transitions: list[Transition] = []
        current = start
        for a in actions:
            trans = self.simulate(current, a)
            transitions.append(trans)
            if not trans.succeeded:
                break
            current = trans.to_state
        return transitions

    def plan_is_feasible(
        self, start: TypedState, actions: list[str]
    ) -> bool:
        transitions = self.simulate_sequence(start, actions)
        return len(transitions) == len(actions) and all(
            t.succeeded for t in transitions
        )

    # ------------------------------------------------------------------
    # Counterfactual estimation (spec §8.3)
    # ------------------------------------------------------------------
    def counterfactual_sequences(
        self,
        start: TypedState,
        candidates: list[list[str]],
        max_candidates: Optional[int] = None,
    ) -> list[tuple[list[str], bool, int]]:
        """Return (sequence, feasible?, estimated_latency_ms) for each
        candidate action sequence. Bounded by `max_candidates` — the
        Scheduler's planning budget caps how many alternatives we try."""
        limit = (
            min(max_candidates, len(candidates))
            if max_candidates is not None
            else len(candidates)
        )
        out: list[tuple[list[str], bool, int]] = []
        for seq in candidates[:limit]:
            feasible = self.plan_is_feasible(start, seq)
            latency = sum(
                (self._schemas[a].estimated_latency_ms if a in self._schemas else 0)
                for a in seq
            )
            out.append((seq, feasible, latency))
        return out
