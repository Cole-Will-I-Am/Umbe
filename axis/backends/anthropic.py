"""Anthropic backend — production-grade wrapper.

Wraps the official ``anthropic`` Python SDK behind the InferenceBackend
Protocol so any AxisRuntime can swap StubBackend -> AnthropicBackend
with a single line.  The SDK is optional — importing this module without
``anthropic`` installed still works; instantiating AnthropicBackend
raises with a clear message.

Production features implemented here:

    * **Prompt caching** — the system prompt is sent with
      ``cache_control={"type": "ephemeral"}`` so repeated calls sharing
      the same system prefix hit the Anthropic prompt cache.  Cache hit
      / creation token counts are surfaced on ``BackendResult``.

    * **Tool-use wiring** — ``tool_inventory`` is converted to the SDK's
      ``tools`` parameter format. ``tool_use`` blocks in the response are
      extracted and surfaced as ``BackendResult.tool_calls``.

    * **Heuristic entropy estimation** — until the API exposes per-token
      logprobs, we compute a lightweight proxy from response properties
      (hedging markers, length ratio, refusal patterns). This feeds the
      Verifier's confidence signal rather than returning a constant 0.0.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .base import BackendResult

# IMPORTANT: Claude 4.6 is the most capable model family as of this build.
# Default to Sonnet 4.6 for cost/latency; caller can override per-instance.
DEFAULT_MODEL = "claude-sonnet-4-6"

# Hedging phrases that signal model uncertainty.  Each hit contributes a
# small additive bump to the heuristic entropy estimate.
_HEDGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bi(?:'m| am) not (?:sure|certain)\b", re.IGNORECASE),
    re.compile(r"\bit(?:'s| is) (?:possible|unclear|ambiguous)\b", re.IGNORECASE),
    re.compile(r"\bperhaps\b", re.IGNORECASE),
    re.compile(r"\bmight be\b", re.IGNORECASE),
    re.compile(r"\bcould be\b", re.IGNORECASE),
    re.compile(r"\bI think\b", re.IGNORECASE),
    re.compile(r"\bprobably\b", re.IGNORECASE),
    re.compile(r"\bnot entirely clear\b", re.IGNORECASE),
    re.compile(r"\bhard to say\b", re.IGNORECASE),
    re.compile(r"\bI'?m uncertain\b", re.IGNORECASE),
)

# Refusal / can't-answer patterns.
_REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bI (?:can't|cannot|am unable to)\b", re.IGNORECASE),
    re.compile(r"\bI don'?t have (?:enough|sufficient)\b", re.IGNORECASE),
    re.compile(r"\bbeyond (?:my|the) (?:scope|ability)\b", re.IGNORECASE),
)


class AnthropicBackend:
    """InferenceBackend wrapper around the Anthropic SDK.

    Usage::

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

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------
    def run(
        self,
        prompt: str,
        max_tokens: int,
        tool: Optional[str] = None,
        tool_inventory: Optional[dict] = None,
    ) -> BackendResult:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        # --- Prompt caching: mark the system block as cacheable. ------
        if self.system:
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": self.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # --- Tool-use wiring: convert tool_inventory -> SDK tools. ----
        tools = _build_tools(tool_inventory, tool)
        if tools:
            kwargs["tools"] = tools

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
        cache_read, cache_creation = _extract_cache_tokens(resp)
        entropy = _estimate_entropy(text, max_tokens)

        return BackendResult(
            output=text,
            tokens_used=min(tokens or len(text) // 4, max_tokens),
            success=True,
            error=None,
            entropy=entropy,
            tool_calls=tool_calls,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
        )


# ------------------------------------------------------------------
# Tool inventory conversion
# ------------------------------------------------------------------

def _build_tools(
    tool_inventory: Optional[dict],
    requested_tool: Optional[str] = None,
) -> list[dict]:
    """Convert AXIS tool_inventory to the Anthropic SDK ``tools`` format.

    Each key in *tool_inventory* is a tool name.  The value is a dict
    that may contain:

    * ``description`` (str) — what the tool does.
    * ``parameters`` (dict) — JSON Schema for the input.

    If only *requested_tool* is given (no inventory), we build a
    minimal tool definition so the model knows a tool is expected.
    """
    if not tool_inventory and not requested_tool:
        return []

    tools: list[dict] = []
    seen: set[str] = set()

    if tool_inventory:
        for name, spec in tool_inventory.items():
            tool_def: dict[str, Any] = {
                "name": name,
                "description": spec.get("description", f"Tool: {name}"),
                "input_schema": spec.get("parameters", {"type": "object", "properties": {}}),
            }
            tools.append(tool_def)
            seen.add(name)

    # If the caller asked for a specific tool that wasn't in the
    # inventory, add a minimal stub so the model can still use it.
    if requested_tool and requested_tool not in seen:
        tools.append(
            {
                "name": requested_tool,
                "description": f"Tool: {requested_tool}",
                "input_schema": {"type": "object", "properties": {}},
            }
        )

    return tools


# ------------------------------------------------------------------
# Response extraction helpers
# ------------------------------------------------------------------

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


def _extract_cache_tokens(resp: Any) -> tuple[int, int]:
    """Return (cache_read_tokens, cache_creation_tokens) from the response."""
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return 0, 0

    if isinstance(usage, dict):
        read = usage.get("cache_read_input_tokens", 0)
        creation = usage.get("cache_creation_input_tokens", 0)
    else:
        read = getattr(usage, "cache_read_input_tokens", 0)
        creation = getattr(usage, "cache_creation_input_tokens", 0)

    return int(read or 0), int(creation or 0)


# ------------------------------------------------------------------
# Heuristic entropy estimation
# ------------------------------------------------------------------

def _estimate_entropy(text: str, max_tokens: int) -> float:
    """Produce a [0.0, 1.0] heuristic entropy proxy from response text.

    This is a stand-in until the Anthropic API exposes per-token
    logprobs. The signal is coarse but meaningfully better than the
    previous constant 0.0 — it distinguishes confident answers from
    hedged, short, or refused ones.

    Components:
        1. **Hedging density** — fraction of hedge-pattern hits relative
           to a saturation cap. More hedging → higher entropy.
        2. **Length ratio** — very short responses relative to
           ``max_tokens`` often signal low confidence or refusal.
        3. **Refusal signal** — explicit "I can't" / "I don't have
           enough information" bumps entropy toward the ceiling.

    The final value is clamped to [0.0, 1.0].
    """
    if not text:
        return 1.0  # Empty response is maximally uncertain.

    # 1. Hedging density (0.0 – 0.5 contribution).
    hedge_hits = sum(1 for p in _HEDGE_PATTERNS if p.search(text))
    hedge_score = min(hedge_hits / 6.0, 1.0) * 0.5

    # 2. Length ratio (0.0 – 0.2 contribution).
    # Very short answers relative to budget hint at uncertainty or
    # inability. Ratio is inverted: short → high score.
    word_count = len(text.split())
    expected_words = max(max_tokens * 0.6, 1)  # rough tokens-to-words
    length_score = max(0.0, 1.0 - word_count / expected_words) * 0.2

    # 3. Refusal signal (0.0 or 0.3 contribution).
    refusal_hit = any(p.search(text) for p in _REFUSAL_PATTERNS)
    refusal_score = 0.3 if refusal_hit else 0.0

    entropy = hedge_score + length_score + refusal_score
    return min(max(entropy, 0.0), 1.0)
