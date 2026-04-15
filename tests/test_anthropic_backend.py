"""Anthropic backend — exercised via an injected mock client so the
test suite doesn't need network access or the anthropic SDK installed."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from axis.backends.anthropic import AnthropicBackend, DEFAULT_MODEL


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _ToolBlock:
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class _Resp:
    content: list
    usage: _Usage


def test_run_returns_text_and_tokens():
    client = MagicMock()
    client.messages.create.return_value = _Resp(
        content=[_TextBlock(text="hello from stubbed claude")],
        usage=_Usage(input_tokens=10, output_tokens=7),
    )
    backend = AnthropicBackend(client=client)
    result = backend.run(prompt="hi", max_tokens=100)
    assert result.success
    assert "hello" in result.output
    assert result.tokens_used == 7


def test_run_enforces_max_tokens_cap():
    client = MagicMock()
    client.messages.create.return_value = _Resp(
        content=[_TextBlock(text="hello from stubbed claude")],
        usage=_Usage(input_tokens=10, output_tokens=9999),
    )
    backend = AnthropicBackend(client=client)
    result = backend.run(prompt="hi", max_tokens=50)
    assert result.tokens_used == 50


def test_run_surfaces_tool_use_blocks():
    client = MagicMock()
    client.messages.create.return_value = _Resp(
        content=[
            _TextBlock(text="calling tool"),
            _ToolBlock(name="calculator", input={"expr": "2+2"}),
        ],
        usage=_Usage(input_tokens=5, output_tokens=4),
    )
    backend = AnthropicBackend(client=client)
    result = backend.run(prompt="math", max_tokens=100, tool="calculator")
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["tool"] == "calculator"
    assert result.tool_calls[0]["args"] == {"expr": "2+2"}


def test_run_returns_error_on_exception():
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("api down")
    backend = AnthropicBackend(client=client)
    result = backend.run(prompt="hi", max_tokens=100)
    assert not result.success
    assert "api down" in result.error


def test_default_model_is_claude_4_6():
    """AXIS defaults to the latest Claude family. Update this test when
    the knowledge cutoff advances."""
    client = MagicMock()
    client.messages.create.return_value = _Resp(
        content=[_TextBlock(text="ok")],
        usage=_Usage(input_tokens=1, output_tokens=1),
    )
    backend = AnthropicBackend(client=client)
    assert backend.model == DEFAULT_MODEL
    assert "claude" in DEFAULT_MODEL
    backend.run(prompt="x", max_tokens=10)
    call = client.messages.create.call_args
    assert call.kwargs["model"] == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Tool-use loop + prompt caching
# ---------------------------------------------------------------------------
@dataclass
class _RespWithStop:
    content: list
    usage: _Usage
    stop_reason: str = "end_turn"


def _tool_inventory_with_runner(calls_log: list):
    def _calc(args):
        calls_log.append(args)
        expr = args.get("expr", "")
        return {"result": sum(int(x) for x in expr.split("+"))}

    return {
        "calculator": {
            "description": "Adds integers",
            "input_schema": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
            "run": _calc,
        }
    }


def test_tool_inventory_is_translated_to_tools_param():
    client = MagicMock()
    # Single-shot call (no tool_use in response) — we just want to
    # verify that tools kwarg was populated and the schema preserved.
    client.messages.create.return_value = _RespWithStop(
        content=[_TextBlock(text="answer")],
        usage=_Usage(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    inventory = _tool_inventory_with_runner([])
    backend = AnthropicBackend(client=client)
    backend.run(prompt="add 1+2", max_tokens=100, tool_inventory=inventory)
    call = client.messages.create.call_args
    tools = call.kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "calculator"
    assert tools[0]["description"] == "Adds integers"
    assert tools[0]["input_schema"]["required"] == ["expr"]


def test_tool_use_loop_executes_tool_and_feeds_result_back():
    client = MagicMock()
    first = _RespWithStop(
        content=[
            _TextBlock(text="let me compute"),
            _ToolBlock(name="calculator", input={"expr": "1+2+3"}),
        ],
        usage=_Usage(input_tokens=5, output_tokens=4),
        stop_reason="tool_use",
    )
    # Give the tool_use block an id attribute so the backend can
    # serialise it back into the follow-up request.
    first.content[1].id = "tool_use_1"  # type: ignore[attr-defined]

    second = _RespWithStop(
        content=[_TextBlock(text="the answer is 6")],
        usage=_Usage(input_tokens=2, output_tokens=3),
        stop_reason="end_turn",
    )
    client.messages.create.side_effect = [first, second]

    calls_log: list = []
    inventory = _tool_inventory_with_runner(calls_log)
    backend = AnthropicBackend(client=client)
    result = backend.run(
        prompt="what is 1+2+3?", max_tokens=200, tool_inventory=inventory
    )

    assert result.success
    assert result.output == "the answer is 6"
    assert client.messages.create.call_count == 2
    # Tool was actually executed with the model's args.
    assert calls_log == [{"expr": "1+2+3"}]
    # Audit trail recorded.
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["tool"] == "calculator"
    assert result.tool_calls[0]["result"] == {"result": 6}

    # The second request must include a user turn with a tool_result
    # block keyed by the same tool_use_id.
    second_call = client.messages.create.call_args_list[1]
    messages = second_call.kwargs["messages"]
    # [user, assistant(tool_use), user(tool_result)]
    assert len(messages) == 3
    tool_result_msg = messages[-1]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["type"] == "tool_result"
    assert tool_result_msg["content"][0]["tool_use_id"] == "tool_use_1"


def test_tool_use_loop_respects_max_tool_iters():
    client = MagicMock()

    def _always_tool(*a, **kw):
        block = _ToolBlock(name="calculator", input={"expr": "1+1"})
        block.id = "loop_id"  # type: ignore[attr-defined]
        return _RespWithStop(
            content=[block],
            usage=_Usage(input_tokens=1, output_tokens=1),
            stop_reason="tool_use",
        )

    client.messages.create.side_effect = _always_tool
    backend = AnthropicBackend(client=client, max_tool_iters=3)
    inventory = _tool_inventory_with_runner([])
    result = backend.run(prompt="x", max_tokens=50, tool_inventory=inventory)
    assert not result.success
    assert "max_tool_iters_exceeded" in (result.error or "")
    # Exactly max_tool_iters requests went out.
    assert client.messages.create.call_count == 3


def test_tool_use_loop_records_failing_tool():
    def _boom(_args):
        raise ValueError("bad input")

    inventory = {
        "broken": {
            "description": "always fails",
            "input_schema": {"type": "object"},
            "run": _boom,
        }
    }

    client = MagicMock()
    tool_block = _ToolBlock(name="broken", input={})
    tool_block.id = "err_id"  # type: ignore[attr-defined]
    first = _RespWithStop(
        content=[tool_block],
        usage=_Usage(input_tokens=1, output_tokens=1),
        stop_reason="tool_use",
    )
    second = _RespWithStop(
        content=[_TextBlock(text="giving up")],
        usage=_Usage(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    client.messages.create.side_effect = [first, second]

    backend = AnthropicBackend(client=client)
    result = backend.run(prompt="x", max_tokens=50, tool_inventory=inventory)
    assert result.success
    assert result.tool_calls[0]["success"] is False
    assert "bad input" in (result.tool_calls[0]["error"] or "")
    # The follow-up user message's tool_result must be flagged is_error.
    follow_up = client.messages.create.call_args_list[1].kwargs["messages"][-1]
    assert follow_up["content"][0]["is_error"] is True


def test_prompt_caching_wraps_system_and_tools():
    client = MagicMock()
    client.messages.create.return_value = _RespWithStop(
        content=[_TextBlock(text="ok")],
        usage=_Usage(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    inventory = _tool_inventory_with_runner([])
    backend = AnthropicBackend(
        client=client,
        system="be helpful",
        cache_system=True,
        cache_tools=True,
    )
    backend.run(prompt="x", max_tokens=50, tool_inventory=inventory)

    call = client.messages.create.call_args
    system = call.kwargs["system"]
    assert isinstance(system, list)
    assert system[0]["type"] == "text"
    assert system[0]["cache_control"] == {"type": "ephemeral"}

    tools = call.kwargs["tools"]
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}


def test_prompt_caching_can_be_disabled():
    client = MagicMock()
    client.messages.create.return_value = _RespWithStop(
        content=[_TextBlock(text="ok")],
        usage=_Usage(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    backend = AnthropicBackend(
        client=client, system="plain system", cache_system=False
    )
    backend.run(prompt="x", max_tokens=10)
    call = client.messages.create.call_args
    # With caching off, system falls back to the raw string form.
    assert call.kwargs["system"] == "plain system"


def test_unknown_tool_name_returns_structured_error():
    client = MagicMock()
    block = _ToolBlock(name="ghost", input={})
    block.id = "ghost_id"  # type: ignore[attr-defined]
    first = _RespWithStop(
        content=[block],
        usage=_Usage(input_tokens=1, output_tokens=1),
        stop_reason="tool_use",
    )
    second = _RespWithStop(
        content=[_TextBlock(text="stopping")],
        usage=_Usage(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    client.messages.create.side_effect = [first, second]

    # Registered inventory has a real tool so the loop engages, but the
    # model asks for a different name.
    inventory = _tool_inventory_with_runner([])
    backend = AnthropicBackend(client=client)
    result = backend.run(prompt="x", max_tokens=50, tool_inventory=inventory)
    assert result.success
    assert result.tool_calls[0]["tool"] == "ghost"
    assert result.tool_calls[0]["success"] is False
    assert result.tool_calls[0]["error"] == "unknown_tool"
