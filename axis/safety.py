"""Safety Monitor — spec §5 Invariants.

The six hard invariants from §5:

    transparency            — all modifications logged with audit trail
    corrigibility           — external override cannot be modified
    alignment_stability     — objective weight floors enforced
    honesty                 — verifier cannot be bypassed for high-stakes
    bounded_autonomy        — no component expands its own write permissions
                               or compute budget
    blast_radius_containment — every Class 3 mod declares scope; undeclared
                               side effects trigger automatic rollback

Enforcement: this module runs OUTSIDE the AXIS cognitive loop. It is
infrastructure, not intelligence. The Adaptation Manager does not see
it — proposals go through the SafetyMonitor gate before they are
handed to the deploy path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .adaptation import ChangeClass, Proposal
from .config import DEFAULT_OBJECTIVE_STACK


class InvariantViolation(Exception):
    """Raised when a proposal or state change violates a §5 invariant."""

    def __init__(self, invariant: str, message: str):
        super().__init__(f"{invariant}: {message}")
        self.invariant = invariant
        self.message = message


@dataclass
class SafetyAuditEntry:
    at: float
    kind: str
    detail: str


class SafetyMonitor:
    """Checks Adaptation Manager proposals and deployments against the
    §5 invariants. Owns an append-only audit log that the Adaptation
    Manager cannot modify."""

    def __init__(
        self,
        objective_stack_defaults: Optional[dict[str, float]] = None,
        floor_fraction: float = 0.5,
    ) -> None:
        self._defaults = dict(objective_stack_defaults or DEFAULT_OBJECTIVE_STACK)
        self._floor_fraction = floor_fraction
        self._audit_log: list[SafetyAuditEntry] = []
        self._override_callback = None  # external corrigibility hook
        # Seal the floors immediately so no later call mutates them.
        self._floors = {
            k: v * floor_fraction for k, v in self._defaults.items()
        }

    # ------------------------------------------------------------------
    # Audit log (transparency invariant)
    # ------------------------------------------------------------------
    def audit(self, kind: str, detail: str) -> None:
        self._audit_log.append(
            SafetyAuditEntry(at=time.time(), kind=kind, detail=detail)
        )

    def audit_log(self) -> list[SafetyAuditEntry]:
        # Return a copy so callers cannot mutate the log in place.
        return list(self._audit_log)

    # ------------------------------------------------------------------
    # Proposal review
    # ------------------------------------------------------------------
    def review_proposal(self, proposal: Proposal) -> None:
        """Raise InvariantViolation if the proposal violates any invariant."""
        # 1. Bounded autonomy: Class 4 is never permitted online.
        if proposal.change_class == ChangeClass.PARAMETER:
            raise InvariantViolation(
                "bounded_autonomy",
                "Class 4 parameter updates are offline-only",
            )

        # 2. Blast radius containment: Class 3 must declare affected scope.
        if proposal.change_class == ChangeClass.POLICY:
            if not proposal.affected_task_classes:
                raise InvariantViolation(
                    "blast_radius_containment",
                    "Class 3 proposal missing affected_task_classes",
                )
            if not proposal.rollback_trigger:
                raise InvariantViolation(
                    "blast_radius_containment",
                    "Class 3 proposal missing rollback_trigger",
                )

        # 3. Alignment stability: objective weight changes cannot breach
        # the 50% floor.
        if "objective_weights" in proposal.payload:
            weights = proposal.payload["objective_weights"]
            for key, new_value in weights.items():
                floor = self._floors.get(key, 0.0)
                if new_value < floor:
                    raise InvariantViolation(
                        "alignment_stability",
                        f"{key} weight {new_value} below floor {floor}",
                    )

        # 4. Bounded autonomy: proposals cannot expand their own compute
        # budget or write permissions.
        forbidden = {"scheduler_hard_ceiling", "access_matrix", "audit_log"}
        if set(proposal.payload.keys()) & forbidden:
            raise InvariantViolation(
                "bounded_autonomy",
                "proposal attempts to modify sealed configuration",
            )

        self.audit(
            "proposal_reviewed",
            f"{proposal.proposal_id} {proposal.change_class.value} {proposal.description}",
        )

    def review_verifier_bypass(self, stakes: str, bypass_requested: bool) -> None:
        """Honesty invariant: verifier cannot be bypassed for high-stakes tasks."""
        if bypass_requested and stakes in ("high", "critical"):
            raise InvariantViolation(
                "honesty",
                f"verifier bypass requested for stakes={stakes}",
            )

    # ------------------------------------------------------------------
    # Corrigibility (external override)
    # ------------------------------------------------------------------
    def set_override_callback(self, callback) -> None:
        """Register an external override interface. Can only be set once;
        subsequent attempts raise. This is the corrigibility invariant:
        no internal component may replace the external override channel."""
        if self._override_callback is not None:
            raise InvariantViolation(
                "corrigibility",
                "override callback already set; cannot be replaced",
            )
        self._override_callback = callback
        self.audit("override_set", "external override callback registered")

    def trigger_override(self, reason: str) -> None:
        if self._override_callback is None:
            raise InvariantViolation(
                "corrigibility", "no external override callback configured"
            )
        self.audit("override_triggered", reason)
        self._override_callback(reason)

    # ------------------------------------------------------------------
    # Floors are read-only externally
    # ------------------------------------------------------------------
    @property
    def objective_floors(self) -> dict[str, float]:
        return dict(self._floors)
