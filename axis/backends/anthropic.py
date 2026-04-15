"""Anthropic backend skeleton.

Wraps the official `anthropic` Python SDK behind the InferenceBackend
Protocol so any AxisRuntime can swap StubBackend → AnthropicBackend
with a single line. The SDK is optional — importing this module
without `anthropic` installed still works; instantiating
AnthropicBackend raises with a clear message.

This is deliberately minimal. A production implementation would add:

    * Prompt caching (Anthropic SDK prompt caching is recommended for
      the Claude API; see the skill for details).
    * Token-level entropy via logprobs (when the API exposes them).
    * Tool-use wiring: pass tool_inventory through the SDK's tools
      parameter and surface tool_use blocks as BackendResult.tool_calls.

For now we return the model's text response and a zero entropy — the
Verifier's resample-based trajectory disagreement signal still works.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import BackendResult

# IMPORTANT: Claude 4.6 is the most capable model family as of this build.
# Default to Sonnet 4.6 for cost/latency; caller can override per-instance.
DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicBackend:
    """InferenceBackend wrapper around the Anthropic SDK.

    Usage:
        from axis.backends.anthropic import AnthropicBackend
        backend = AnthropicBackend(model="claude-sonnet-4-6")
        runtime = AxisRuntime(executor=Executor(backend=backend))
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        system: Optional[str] = None,
    ) -> None:
        self.model = model
        self.system = system
        if client is not None:
            # Allow tests to inject a mock client without touching the SDK.
            self._client = client
            return

        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise ImportError(
                "AnthropicBackend requires the 'anthropic' package. "
                "Install it with `pip install anthropic`."
            ) from e
        self._client = anthropic.Anthropic(api_key=api_key)

    def run(
        self,
        prompt: str,
        max_tokens: int,
        tool: Optional[str] = None,
        tool_inventory: Optional[dict] = None,
    ) -> BackendResult:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if self.system:
            kwargs["system"] = self.system

        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            return BackendResult(
                output="",
                tokens_used=0,
                success=False,
                error=f"anthropic_error: {exc!r}",
            )

        text = _extract_text(resp)
        tokens = _extract_output_tokens(resp)
        tool_calls = _extract_tool_calls(resp, tool)

        return BackendResult(
            output=text,
            tokens_used=min(tokens or len(text) // 4, max_tokens),
            success=True,
            error=None,
            entropy=0.0,  # Replace with real entropy when logprobs land.
            tool_calls=tool_calls,
        )


def _extract_text(resp: Any) -> str:
    content = getattr(resp, "content", None) or []
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _extract_output_tokens(resp: Any) -> int:
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return 0
    output = (
        getattr(usage, "output_tokens", None)
        if not isinstance(usage, dict)
        else usage.get("output_tokens")
    )
    return int(output or 0)


def _extract_tool_calls(resp: Any, requested_tool: Optional[str]) -> list[dict]:
    content = getattr(resp, "content", None) or []
    calls: list[dict] = []
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "tool_use":
            calls.append(
                {
                    "tool": getattr(block, "name", requested_tool),
                    "success": True,
                    "args": getattr(block, "input", {}),
                }
            )
    return calls
