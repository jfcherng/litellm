"""
Regression tests for issue #34692.

ollama_chat streams tool_calls in a mid-stream chunk while its final
(``done: true``) chunk carries only ``done_reason: "stop"``. The provider
iterator must remember the earlier tool_calls and stamp
``finish_reason="tool_calls"`` on the final chunk, so the Anthropic
``/v1/messages`` bridge emits ``stop_reason: "tool_use"``. Before the fix the
bridge emitted ``stop_reason: "end_turn"`` and Anthropic tool-runners
(Claude Code, ``messages.stream``) silently dropped the tool call.
"""

from typing import Final

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.llms.ollama.chat.transformation import (
    OllamaChatCompletionResponseIterator,
)
from litellm.types.utils import ModelResponseStream

_OLLAMA_TOOL_CHUNK = {
    "model": "qwen3:8b",
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "San Francisco"}}}],
    },
    "done": False,
}
_OLLAMA_DONE_CHUNK = {
    "model": "qwen3:8b",
    "message": {"role": "assistant", "content": ""},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 100,
    "eval_count": 20,
}


def _ollama_streamed_chunks() -> list[ModelResponseStream]:
    iterator = OllamaChatCompletionResponseIterator(streaming_response=iter([]), sync_stream=True)
    return [iterator.chunk_parser(_OLLAMA_TOOL_CHUNK), iterator.chunk_parser(_OLLAMA_DONE_CHUNK)]


class _AsyncStream:
    def __init__(self, items: list[ModelResponseStream]):
        self._it = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


def _assert_tool_use_stop_reason(events: list[dict]) -> None:
    block_types = [e["content_block"]["type"] for e in events if e.get("type") == "content_block_start"]
    assert "tool_use" in block_types, f"no tool_use content block opened: {events}"
    message_deltas = [e for e in events if e.get("type") == "message_delta"]
    assert message_deltas, f"no message_delta emitted: {events}"
    assert message_deltas[-1]["delta"]["stop_reason"] == "tool_use", (
        f"expected stop_reason 'tool_use', got: {message_deltas[-1]}"
    )


def test_ollama_mid_stream_tool_call_yields_tool_use_stop_reason_sync():
    wrapper = AnthropicStreamWrapper(completion_stream=iter(_ollama_streamed_chunks()), model="qwen3:8b")
    _assert_tool_use_stop_reason(list(wrapper))


@pytest.mark.asyncio
async def test_ollama_mid_stream_tool_call_yields_tool_use_stop_reason_async():
    wrapper = AnthropicStreamWrapper(completion_stream=_AsyncStream(_ollama_streamed_chunks()), model="qwen3:8b")
    _assert_tool_use_stop_reason([event async for event in wrapper])


_CHUNK_BASE: Final = {"id": "chatcmpl-t", "object": "chat.completion.chunk", "created": 1, "model": "expert"}
_TOOL_ARGUMENTS: Final = ('{"path": "/tmp/x.py", ', '"content": "hi"}')
_USAGE: Final = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


def _chunk(delta: dict[str, object], finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(**_CHUNK_BASE, choices=[{"index": 0, "delta": delta, "finish_reason": finish_reason}])


def _usage_chunk() -> ModelResponseStream:
    return ModelResponseStream(**_CHUNK_BASE, choices=[], usage=_USAGE)


def _arguments_chunk(arguments: str, finish_reason: str | None = None) -> ModelResponseStream:
    return _chunk({"tool_calls": [{"index": 0, "function": {"arguments": arguments}}]}, finish_reason)


def _tool_call_opener() -> ModelResponseStream:
    opener = {"index": 0, "id": "chatcmpl-tool-abc", "type": "function", "function": {"name": "write", "arguments": ""}}
    return _chunk({"tool_calls": [opener]})


def _events(chunks: list[ModelResponseStream]) -> list[dict]:
    return list(AnthropicStreamWrapper(completion_stream=iter(chunks), model="expert"))


def _stop_reason(events: list[dict]) -> str:
    return [event for event in events if event["type"] == "message_delta"][-1]["delta"]["stop_reason"]


def _tool_arguments(events: list[dict]) -> str:
    return "".join(
        event["delta"]["partial_json"]
        for event in events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "input_json_delta"
    )


def test_stop_finish_reason_after_tool_call_reports_tool_use():
    chunks = [
        _chunk({"role": "assistant"}),
        _tool_call_opener(),
        _arguments_chunk(_TOOL_ARGUMENTS[0]),
        _arguments_chunk(_TOOL_ARGUMENTS[1], finish_reason="stop"),
        _usage_chunk(),
    ]

    events = _events(chunks)

    assert _tool_arguments(events) == "".join(_TOOL_ARGUMENTS)
    assert _stop_reason(events) == "tool_use"


def test_truncated_tool_call_keeps_max_tokens_stop_reason():
    chunks = [
        _chunk({"role": "assistant"}),
        _tool_call_opener(),
        _arguments_chunk(_TOOL_ARGUMENTS[0]),
        _chunk({}, finish_reason="length"),
        _usage_chunk(),
    ]

    assert _stop_reason(_events(chunks)) == "max_tokens"
