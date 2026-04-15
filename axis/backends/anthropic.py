"""Anthropic backend — real tool use + prompt caching.

Wraps the official ``anthropic`` Python SDK behind the InferenceBackend
Protocol so any AxisRuntime can swap StubBackend → AnthropicBackend
with a single line. The SDK is optional — importing this module
without ``anthropic`` installed still works; instantiating
AnthropicBackend raises with a clear message.

This build implements three things the old skeleton flagged as TODO:

1. **Tool use**: when the caller supplies a non-empty ``tool_inventory``
   mapping tool names to executable specs, the backend translates them
   to the SDK's ``tools=[...]`` parameter, loops over ``tool_use`` turns,
   executes each registered callable, feeds results back as
   ``tool_result`` messages, and returns the final text plus an audit
   trail in ``BackendResult.tool_calls``. Bounded by ``max_tool_iters``.

2. **Prompt caching**: the system prompt and tool schema (the two large,
   stable prefixes of every agent-loop request) are wrapped with
   ``cache_control: {type: ephemeral}`` so repeated calls with the same
   system + tools share a cached prefix. Opt-out via constructor flags.

3. **Preserves the old behavior**: if no ``tool_inventory`` is passed,
   or if the inventory values are not callable tool specs, the backend
   still works the way it did before — single request, plain text
   response, legacy ``_extract_tool_calls`` fallback.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Union

from .base import BackendResult

# IMPORTANT: Claude 4.6 is the most capable model family as of this build.
# Default to Sonnet 4.6 for cost/latency; caller can override per-instance.
DEFAULT_MODEL = "claude-sonnet-4-6"

# A tool entry in ``tool_inventory`` can be either:
#   (a) a ``ToolSpec`` dict with {"description", "input_schema", "run"}
#   (b) a bare callable — treated as a zero-schema tool
# This dual form keeps the old Executor call sites (which only know a
# ``tool=`` name) working while letting real agent loops register
# executable tools.
ToolSpec = Mapping[str, Any]
ToolEntry = Union[ToolSpec, Callable[[dict], Any]]


class AnthropicBackend:
    """InferenceBackend wrapper around the Anthropic SDK.

    Usage:

        from axis.backends.anthropic import AnthropicBackend
        backend = AnthropicBackend(
            model="claude-sonnet-4-6",
            system="You are AXIS's execution engine.",
        )
        runtime = AxisRuntime(executor=Executor(backend=backend))

    Tool-use example::

        def _calculator(args):
            return {"result": eval(args["expr"], {"__builtins__": {}})}

        inventory = {
            "calculator": {
                "description": "Evaluate a math expression.",
                "input_schema": {
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                    "required": ["expr"],
                },
                "run": _calculator,
            }
        }
        result = backend.run(
            prompt="What is 2+2*3?",
            max_tokens=512,
            tool_inventory=inventory,
        )
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        system: Optional[str] = None,
        cache_system: bool = True,
        cache_tools: bool = True,
        max_tool_iters: int = 5,
    ) -> None:
        self.model = model
        self.system = system
        self.cache_system = cache_system
        self.cache_tools = cache_tools
        self.max_tool_iters = max_tool_iters
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
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        prompt: str,
        max_tokens: int,
        tool: Optional[str] = None,
        tool_inventory: Optional[dict] = None,
    ) -> BackendResult:
        messages: list[dict] = [{"role": "user", "content": prompt}]
        tools_param = self._tools_param(tool_inventory)

        # If the caller supplied executable tools, run a full tool-use
        # loop. Otherwise fall through to a single-shot call — the
        # legacy path — so existing tests and non-agent code keep
        # working verbatim.
        if tools_param and _has_executable_tools(tool_inventory):
            return self._run_tool_loop(
                messages=messages,
                tools_param=tools_param,
                tool_inventory=tool_inventory or {},
                max_tokens=max_tokens,
                requested_tool=tool,
            )

        return self._single_call(
            messages=messages,
            tools_param=tools_param,
            max_tokens=max_tokens,
            requested_tool=tool,
        )

    # ------------------------------------------------------------------
    # Single-call path (legacy + fallback)
    # ------------------------------------------------------------------
    def _single_call(
        self,
        messages: list[dict],
        tools_param: Optional[list[dict]],
        max_tokens: int,
        requested_tool: Optional[str],
    ) -> BackendResult:
        kwargs = self._base_kwargs(max_tokens=max_tokens, tools_param=tools_param)
        kwargs["messages"] = messages

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
        tool_calls = _extract_tool_calls(resp, requested_tool)

        return BackendResult(
            output=text,
            tokens_used=min(tokens or len(text) // 4, max_tokens),
            success=True,
            error=None,
            entropy=0.0,  # Replace with real entropy when logprobs land.
            tool_calls=tool_calls,
        )

    # ------------------------------------------------------------------
    # Tool-use loop
    # ------------------------------------------------------------------
    def _run_tool_loop(
        self,
        messages: list[dict],
        tools_param: list[dict],
        tool_inventory: dict,
        max_tokens: int,
        requested_tool: Optional[str],
    ) -> BackendResult:
        """Run bounded model ↔ tool turns until the model stops calling
        tools or we hit the iteration cap. Each tool invocation is
        recorded in ``BackendResult.tool_calls`` for audit / telemetry.
        """
        all_tool_calls: list[dict] = []
        total_output_tokens = 0
        final_text = ""

        for iteration in range(self.max_tool_iters):
            kwargs = self._base_kwargs(
                max_tokens=max_tokens, tools_param=tools_param
            )
            kwargs["messages"] = messages
            try:
                resp = self._client.messages.create(**kwargs)
            except Exception as exc:
                return BackendResult(
                    output=final_text,
                    tokens_used=min(total_output_tokens, max_tokens),
                    success=False,
                    error=f"anthropic_error: {exc!r}",
                    tool_calls=all_tool_calls,
                )

            total_output_tokens += _extract_output_tokens(resp)
            final_text = _extract_text(resp)
            content_blocks = _response_content(resp)
            tool_uses = [b for b in content_blocks if _block_type(b) == "tool_use"]

            stop_reason = _stop_reason(resp)
            if not tool_uses or stop_reason != "tool_use":
                # Done — the model produced its final answer.
                return BackendResult(
                    output=final_text,
                    tokens_used=min(total_output_tokens, max_tokens),
                    success=True,
                    tool_calls=all_tool_calls,
                )

            # Append the assistant's raw content (including tool_use
            # blocks) to the conversation so the next turn can attach
            # tool_result blocks keyed by tool_use_id.
            messages.append(
                {"role": "assistant", "content": _serialise_content(content_blocks)}
            )

            tool_result_blocks: list[dict] = []
            for block in tool_uses:
                name = _block_attr(block, "name") or ""
                tool_input = _block_attr(block, "input") or {}
                tool_use_id = _block_attr(block, "id") or ""

                entry = tool_inventory.get(name)
                output, success, error = _invoke_tool(entry, tool_input)

                all_tool_calls.append(
                    {
                        "tool": name,
                        "success": success,
                        "args": tool_input,
                        "result": output,
                        "error": error,
                    }
                )
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": _stringify_tool_result(output, error),
                        "is_error": not success,
                    }
                )

            messages.append({"role": "user", "content": tool_result_blocks})

        # Hit the iteration cap — return whatever the last assistant
        # turn produced with a bounded-autonomy style error.
        return BackendResult(
            output=final_text,
            tokens_used=min(total_output_tokens, max_tokens),
            success=False,
            error=f"max_tool_iters_exceeded:{self.max_tool_iters}",
            tool_calls=all_tool_calls,
        )

    # ------------------------------------------------------------------
    # Request-building helpers
    # ------------------------------------------------------------------
    def _base_kwargs(
        self,
        max_tokens: int,
        tools_param: Optional[list[dict]],
    ) -> dict:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
        }
        if self.system:
            kwargs["system"] = self._maybe_cached_system()
        if tools_param:
            kwargs["tools"] = self._maybe_cached_tools(tools_param)
        return kwargs

    def _maybe_cached_system(self) -> Any:
        if not self.cache_system:
            return self.system
        # Structured form — Anthropic accepts a list of content blocks
        # with cache_control on the last element of the cached prefix.
        return [
            {
                "type": "text",
                "text": self.system or "",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _maybe_cached_tools(self, tools_param: list[dict]) -> list[dict]:
        if not self.cache_tools or not tools_param:
            return tools_param
        # Mark the LAST tool with cache_control so the whole tool prefix
        # becomes a cached block. Anthropic docs recommend putting the
        # breakpoint on the final stable element.
        cached = [dict(t) for t in tools_param]
        cached[-1]["cache_control"] = {"type": "ephemeral"}
        return cached

    def _tools_param(
        self, tool_inventory: Optional[dict]
    ) -> Optional[list[dict]]:
        """Translate an AXIS tool_inventory to Anthropic's tools schema."""
        if not tool_inventory:
            return None
        out: list[dict] = []
        for name, entry in tool_inventory.items():
            description, schema = _tool_description_and_schema(entry)
            out.append(
                {
                    "name": name,
                    "description": description,
                    "input_schema": schema,
                }
            )
        return out or None


# ---------------------------------------------------------------------------
# Response extraction helpers
# ---------------------------------------------------------------------------
def _response_content(resp: Any) -> list[Any]:
    content = getattr(resp, "content", None)
    if content is None and isinstance(resp, dict):
        content = resp.get("content")
    return list(content or [])


def _block_type(block: Any) -> Optional[str]:
    return _block_attr(block, "type")


def _block_attr(block: Any, attr: str) -> Any:
    val = getattr(block, attr, None)
    if val is None and isinstance(block, dict):
        val = block.get(attr)
    return val


def _stop_reason(resp: Any) -> Optional[str]:
    sr = getattr(resp, "stop_reason", None)
    if sr is None and isinstance(resp, dict):
        sr = resp.get("stop_reason")
    return sr


def _extract_text(resp: Any) -> str:
    parts: list[str] = []
    for block in _response_content(resp):
        if _block_type(block) == "text":
            parts.append(_block_attr(block, "text") or "")
        else:
            text = _block_attr(block, "text")
            if text:
                parts.append(text)
    return "".join(parts)


def _extract_output_tokens(resp: Any) -> int:
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return 0
    if isinstance(usage, dict):
        output = usage.get("output_tokens")
    else:
        output = getattr(usage, "output_tokens", None)
    return int(output or 0)


def _extract_tool_calls(resp: Any, requested_tool: Optional[str]) -> list[dict]:
    calls: list[dict] = []
    for block in _response_content(resp):
        if _block_type(block) == "tool_use":
            calls.append(
                {
                    "tool": _block_attr(block, "name") or requested_tool,
                    "success": True,
                    "args": _block_attr(block, "input") or {},
                }
            )
    return calls


def _serialise_content(blocks: list[Any]) -> list[dict]:
    """Re-serialise assistant content blocks into the dict shape the SDK
    accepts on the way back in. Handles both SDK objects and raw dicts.
    """
    out: list[dict] = []
    for block in blocks:
        btype = _block_type(block)
        if btype == "text":
            out.append({"type": "text", "text": _block_attr(block, "text") or ""})
        elif btype == "tool_use":
            out.append(
                {
                    "type": "tool_use",
                    "id": _block_attr(block, "id") or "",
                    "name": _block_attr(block, "name") or "",
                    "input": _block_attr(block, "input") or {},
                }
            )
    return out


# ---------------------------------------------------------------------------
# Tool-entry helpers
# ---------------------------------------------------------------------------
def _has_executable_tools(inventory: Optional[dict]) -> bool:
    if not inventory:
        return False
    for entry in inventory.values():
        if callable(entry):
            return True
        if isinstance(entry, Mapping) and callable(entry.get("run")):
            return True
    return False


def _tool_description_and_schema(entry: ToolEntry) -> tuple[str, dict]:
    if callable(entry):
        return (
            getattr(entry, "__doc__", None) or "",
            {"type": "object", "properties": {}, "additionalProperties": True},
        )
    description = str(entry.get("description", "") or "")
    schema = entry.get("input_schema") or {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }
    return description, dict(schema)


def _invoke_tool(
    entry: Optional[ToolEntry], tool_input: dict
) -> tuple[Any, bool, Optional[str]]:
    if entry is None:
        return None, False, "unknown_tool"
    runner: Optional[Callable[[dict], Any]]
    if callable(entry):
        runner = entry  # type: ignore[assignment]
    else:
        runner = entry.get("run") if isinstance(entry, Mapping) else None
    if runner is None:
        return None, False, "tool_not_executable"
    try:
        result = runner(tool_input)
    except Exception as exc:
        return None, False, f"tool_exception:{exc!r}"
    return result, True, None


def _stringify_tool_result(result: Any, error: Optional[str]) -> str:
    if error:
        return f"ERROR: {error}"
    if isinstance(result, str):
        return result
    try:
        import json

        return json.dumps(result)
    except Exception:
        return str(result)
