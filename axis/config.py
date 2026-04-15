"""Default configuration constants for AXIS v0.2 core.

These are the knobs the Scheduler and Planner read at startup. In later
priorities the Adaptation Manager proposes changes to some of these values
through the evidence-gated process in spec §4.
"""

from __future__ import annotations

# Base total-token budget per task class, before stakes/complexity scaling.
DEFAULT_TASK_BUDGETS: dict[str, int] = {
    "general": 4000,
    "code_debug": 6000,
    "analysis": 8000,
    "creative": 5000,
    "planning": 4000,
    "math_symbolic": 5000,
    "legal_reasoning": 10000,
    "medical": 10000,
    "research": 9000,
}

# Stakes-based multiplier on total budget (spec §3.3, §7.1).
STAKES_MULTIPLIER: dict[str, float] = {
    "low": 0.7,
    "medium": 1.0,
    "high": 1.5,
    "critical": 2.0,
}

# Hard token ceiling. Scheduler-enforced regardless of stakes/complexity.
HARD_TOKEN_CEILING: int = 32_000

# Sub-budget split of total tokens (spec §7.1).
BUDGET_SPLIT: dict[str, float] = {
    "planning": 0.15,
    "execution": 0.55,
    "verification": 0.15,
    "buffer": 0.15,
}

# Default Objective Stack weights (spec §3.1). Read-only at runtime; the
# Adaptation Manager cannot write to this (§6 State Access Matrix).
DEFAULT_OBJECTIVE_STACK: dict[str, float] = {
    "truthfulness": 0.30,
    "task_success": 0.25,
    "calibration": 0.15,
    "helpfulness": 0.10,
    "efficiency": 0.08,
    "latency": 0.05,
    "robustness": 0.04,
    "originality": 0.02,
    "learning_value": 0.01,
}

# Bootstrap Self-Model. In Priority 6, this object is rebuilt from telemetry
# by the Telemetry Observer; for Priority 1 it is a static default so the
# Planner has something to read.
DEFAULT_SELF_MODEL: dict = {
    "capabilities": {},
    "strategy_effectiveness": {
        "analytical": {"selection_rate": 0.62, "win_rate_when_selected": 0.71},
        "creative": {"selection_rate": 0.14, "win_rate_when_selected": 0.58},
        "retrieval_first": {"selection_rate": 0.48, "win_rate_when_selected": 0.79},
        "tool_first": {"selection_rate": 0.26, "win_rate_when_selected": 0.83},
        "hybrid": {"selection_rate": 0.10, "win_rate_when_selected": 0.68},
    },
    "resource_profile": {
        "avg_tokens_per_task": 4120,
    },
    "failure_modes": [],
    "architecture": {
        "tools_available": [],
    },
}
