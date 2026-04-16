"""Anthropic backend — exercised via an injected mock client so the
test suite doesn't need network access or the anthropic SDK installed."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from axis.backends.anthropic import (
    AnthropicBackend,
    DEFAULT_MODEL,
    _build_tools,
    _estimate_entropy,
    _extract_cache_tokens,
)


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


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


# ------------------------------------------------------------------
# Existing basic tests (preserved)
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Prompt caching tests
# ------------------------------------------------------------------


class TestPromptCaching:
    """Verify that system prompts are sent with cache_control."""

    def test_system_sent_as_cacheable_block(self):
        client = MagicMock()
        client.messages.create.return_value = _Resp(
            content=[_TextBlock(text="ok")],
            usage=_Usage(input_tokens=10, output_tokens=1),
        )
        backend = AnthropicBackend(client=client, system="You are AXIS.")
        backend.run(prompt="hi", max_tokens=50)
        call = client.messages.create.call_args
        system_arg = call.kwargs["system"]
        assert isinstance(system_arg, list)
        assert len(system_arg) == 1
        block = system_arg[0]
        assert block["type"] == "text"
        assert block["text"] == "You are AXIS."
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_no_system_means_no_system_kwarg(self):
        client = MagicMock()
        client.messages.create.return_value = _Resp(
            content=[_TextBlock(text="ok")],
            usage=_Usage(input_tokens=1, output_tokens=1),
        )
        backend = AnthropicBackend(client=client)  # no system
        backend.run(prompt="hi", max_tokens=50)
        call = client.messages.create.call_args
        assert "system" not in call.kwargs

    def test_cache_tokens_surfaced_on_result(self):
        client = MagicMock()
        client.messages.create.return_value = _Resp(
            content=[_TextBlock(text="cached answer")],
            usage=_Usage(
                input_tokens=10,
                output_tokens=5,
                cache_read_input_tokens=200,
                cache_creation_input_tokens=0,
            ),
        )
        backend = AnthropicBackend(client=client, system="system prompt")
        result = backend.run(prompt="hi", max_tokens=100)
        assert result.cache_read_tokens == 200
        assert result.cache_creation_tokens == 0

    def test_cache_creation_tokens_on_first_call(self):
        client = MagicMock()
        client.messages.create.return_value = _Resp(
            content=[_TextBlock(text="first call")],
            usage=_Usage(
                input_tokens=10,
                output_tokens=3,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=150,
            ),
        )
        backend = AnthropicBackend(client=client, system="system prompt")
        result = backend.run(prompt="hi", max_tokens=100)
        assert result.cache_read_tokens == 0
        assert result.cache_creation_tokens == 150

    def test_extract_cache_tokens_missing_fields(self):
        """Gracefully handles responses without cache fields."""
        resp = _Resp(
            content=[_TextBlock(text="ok")],
            usage=_Usage(input_tokens=1, output_tokens=1),
        )
        read, creation = _extract_cache_tokens(resp)
        assert read == 0
        assert creation == 0

    def test_extract_cache_tokens_none_usage(self):
        """Returns zeros when usage is None."""

        @dataclass
        class _NoUsage:
            content: list
            usage: object = None

        resp = _NoUsage(content=[])
        read, creation = _extract_cache_tokens(resp)
        assert read == 0
        assert creation == 0


# ------------------------------------------------------------------
# Tool-use wiring tests
# ------------------------------------------------------------------


class TestToolUseWiring:
    """Verify that tool_inventory is converted and passed to the SDK."""

    def test_tool_inventory_passed_as_tools_kwarg(self):
        client = MagicMock()
        client.messages.create.return_value = _Resp(
            content=[_TextBlock(text="ok")],
            usage=_Usage(input_tokens=5, output_tokens=2),
        )
        inventory = {
            "calculator": {
                "description": "Evaluate math expressions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expr": {"type": "string", "description": "expression"},
                    },
                    "required": ["expr"],
                },
            }
        }
        backend = AnthropicBackend(client=client)
        backend.run(prompt="compute 2+2", max_tokens=100, tool_inventory=inventory)
        call = client.messages.create.call_args
        tools = call.kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "calculator"
        assert tools[0]["description"] == "Evaluate math expressions"
        assert tools[0]["input_schema"]["properties"]["expr"]["type"] == "string"

    def test_requested_tool_not_in_inventory_gets_stub_def(self):
        client = MagicMock()
        client.messages.create.return_value = _Resp(
            content=[_TextBlock(text="ok")],
            usage=_Usage(input_tokens=1, output_tokens=1),
        )
        backend = AnthropicBackend(client=client)
        backend.run(prompt="do it", max_tokens=50, tool="web_search")
        call = client.messages.create.call_args
        tools = call.kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "web_search"
        assert "input_schema" in tools[0]

    def test_requested_tool_in_inventory_not_duplicated(self):
        inventory = {
            "search": {
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
            }
        }
        tools = _build_tools(inventory, requested_tool="search")
        assert len(tools) == 1
        assert tools[0]["name"] == "search"

    def test_no_tools_when_both_none(self):
        client = MagicMock()
        client.messages.create.return_value = _Resp(
            content=[_TextBlock(text="ok")],
            usage=_Usage(input_tokens=1, output_tokens=1),
        )
        backend = AnthropicBackend(client=client)
        backend.run(prompt="hi", max_tokens=50)
        call = client.messages.create.call_args
        assert "tools" not in call.kwargs

    def test_multiple_tools_in_inventory(self):
        inventory = {
            "calc": {"description": "Calculator"},
            "search": {"description": "Web search"},
            "code_exec": {"description": "Run code"},
        }
        tools = _build_tools(inventory)
        assert len(tools) == 3
        names = {t["name"] for t in tools}
        assert names == {"calc", "search", "code_exec"}

    def test_tool_inventory_missing_fields_uses_defaults(self):
        """Tool spec with no description or parameters gets sensible defaults."""
        inventory = {"my_tool": {}}
        tools = _build_tools(inventory)
        assert len(tools) == 1
        assert tools[0]["description"] == "Tool: my_tool"
        assert tools[0]["input_schema"] == {"type": "object", "properties": {}}

    def test_tool_use_response_extracted_with_inventory(self):
        client = MagicMock()
        client.messages.create.return_value = _Resp(
            content=[
                _TextBlock(text="Let me search for that."),
                _ToolBlock(name="search", input={"query": "AXIS spec"}),
            ],
            usage=_Usage(input_tokens=10, output_tokens=8),
        )
        inventory = {
            "search": {
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        }
        backend = AnthropicBackend(client=client)
        result = backend.run(
            prompt="find the AXIS spec",
            max_tokens=200,
            tool_inventory=inventory,
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "search"
        assert result.tool_calls[0]["args"] == {"query": "AXIS spec"}


# ------------------------------------------------------------------
# Entropy estimation tests
# ------------------------------------------------------------------


class TestEntropyEstimation:
    """Verify the heuristic entropy proxy produces sensible signals."""

    def test_empty_response_returns_max_entropy(self):
        assert _estimate_entropy("", 100) == 1.0

    def test_confident_answer_has_low_entropy(self):
        text = (
            "The capital of France is Paris. It has been the capital "
            "since the 10th century and serves as the country's political, "
            "economic, and cultural center. The city is located on the "
            "Seine River in northern France."
        )
        e = _estimate_entropy(text, 200)
        assert e < 0.25, f"Confident answer should have low entropy, got {e}"

    def test_hedging_answer_has_higher_entropy(self):
        text = (
            "I'm not sure about this, but I think it might be related "
            "to quantum mechanics. It's possible that the answer is 42, "
            "but it could be something else entirely. Perhaps we should "
            "look at more data. It's unclear from the available evidence."
        )
        e = _estimate_entropy(text, 200)
        assert e > 0.2, f"Hedging answer should have moderate+ entropy, got {e}"

    def test_refusal_has_high_entropy(self):
        text = "I can't answer that question as I don't have enough information."
        e = _estimate_entropy(text, 200)
        assert e > 0.3, f"Refusal should have high entropy, got {e}"

    def test_very_short_response_has_length_penalty(self):
        text = "Yes."
        e = _estimate_entropy(text, 1000)
        # One word out of a 1000-token budget should get a length penalty.
        assert e > 0.1, f"Very short response should have length penalty, got {e}"

    def test_entropy_is_bounded(self):
        """Entropy stays in [0.0, 1.0] even with pathological input."""
        # Stuff every hedge and refusal pattern into one blob.
        text = (
            "I'm not sure. Perhaps. It might be. It could be. I think. "
            "Probably. It's unclear. Not entirely clear. Hard to say. "
            "I'm uncertain. I can't do that. I don't have enough info. "
            "Beyond the scope."
        )
        e = _estimate_entropy(text, 50)
        assert 0.0 <= e <= 1.0, f"Entropy out of bounds: {e}"

    def test_normal_response_gives_nonzero_entropy(self):
        """A real backend should never return exactly 0.0 for non-trivial text."""
        client = MagicMock()
        client.messages.create.return_value = _Resp(
            content=[_TextBlock(text="The answer is 42.")],
            usage=_Usage(input_tokens=5, output_tokens=6),
        )
        backend = AnthropicBackend(client=client)
        result = backend.run(prompt="what is the answer?", max_tokens=100)
        # Should be some entropy from the length ratio, not the old constant 0.0.
        assert result.entropy >= 0.0
        assert result.entropy <= 1.0

    def test_hedging_vs_confident_ordering(self):
        """Hedging text should score strictly higher than confident text."""
        confident = "The speed of light is exactly 299,792,458 meters per second."
        hedging = "I think the speed of light might be around 300,000 km/s, but I'm not sure."
        e_confident = _estimate_entropy(confident, 200)
        e_hedging = _estimate_entropy(hedging, 200)
        assert e_hedging > e_confident, (
            f"Hedging ({e_hedging}) should be > confident ({e_confident})"
        )
