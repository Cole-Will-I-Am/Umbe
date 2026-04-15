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
