"""Deterministic stub backend.

Lets us exercise the Scheduler → Planner → Executor control flow end-to-end
without hitting a real model. Behaviour is fully determined by the prompt
and constructor arguments so tests are reproducible.
"""

from __future__ import annotations

from typing import Optional

from .base import BackendResult


class StubBackend:
    """A fake inference backend for tests and local dry runs.

    * Token count is ~len(prompt)/4, clipped to the caller's max_tokens.
    * If any string in `fail_on` appears in the prompt, the call fails with
      a structured error. This lets tests simulate step failures.
    * Each call is recorded in `self.calls` for assertion.
    """

    def __init__(
        self,
        fail_on: Optional[set[str]] = None,
        fixed_entropy: float = 0.05,
    ):
        self.fail_on: set[str] = set(fail_on) if fail_on else set()
        self.fixed_entropy = fixed_entropy
        self.calls: list[dict] = []

    def run(
        self,
        prompt: str,
        max_tokens: int,
        tool: Optional[str] = None,
        tool_inventory: Optional[dict] = None,
    ) -> BackendResult:
        self.calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "tool": tool,
                "tool_inventory_keys": list((tool_inventory or {}).keys()),
            }
        )

        for trigger in self.fail_on:
            if trigger in prompt:
                return BackendResult(
                    output="",
                    tokens_used=0,
                    success=False,
                    error=f"stub_triggered_failure:{trigger}",
                )

        approx_tokens = max(1, min(max_tokens, len(prompt) // 4 + 32))
        tool_calls: list[dict] = []
        if tool:
            tool_calls.append(
                {
                    "tool": tool,
                    "success": True,
                    "args": {"prompt_preview": prompt[:64]},
                }
            )
        output = f"<stub:{tool or 'think'}> {prompt[:256]}"
        return BackendResult(
            output=output,
            tokens_used=approx_tokens,
            success=True,
            entropy=self.fixed_entropy,
            tool_calls=tool_calls,
        )
