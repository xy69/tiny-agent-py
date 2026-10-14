from __future__ import annotations

import json

import httpx
import pytest

from tiny_agent import Message, ToolCall, ToolDefinition
from tiny_agent.providers import OpenAICompatibleProvider, ProviderError
from tiny_agent.providers.openai import format_messages, format_tools, parse_sse_data
from tiny_agent.types import (
    DoneChunk,
    TextChunk,
    ToolCallDeltaChunk,
    ToolCallStartChunk,
    UsageChunk,
)


def sse(*events: dict[str, object] | str) -> bytes:
    lines = [e if isinstance(e, str) else json.dumps(e) for e in events]
    return "".join(f"data: {ln}\n\n" for ln in lines).encode()


def provider_with(body: bytes, status: int = 200) -> tuple[OpenAICompatibleProvider, list[dict]]:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(status, content=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatibleProvider("m", api_key="k", base_url="http://x/v1", client=client), seen


def test_parse_sse_data_ignores_comments_and_blank_lines() -> None:
    assert parse_sse_data([": keep-alive", "", "data: {}", "event: x"]) == ["{}"]


def test_format_messages_encodes_tool_calls_and_skips_system() -> None:
    msgs = [
        Message("system", "ignored"),
        Message("user", "hi"),
        Message("assistant", "", tool_calls=[ToolCall("c1", "add", {"a": 1})]),
        Message("tool", "2", tool_call_id="c1"),
    ]
    out = format_messages(msgs, "sys")
    assert out[0] == {"role": "system", "content": "sys"}
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[2]["content"] is None
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"a": 1}
    assert out[3]["tool_call_id"] == "c1"


def test_format_tools_wraps_definitions_as_functions() -> None:
    out = format_tools([ToolDefinition("t", "d", {"type": "object"})])
    assert out == [
        {
            "type": "function",
            "function": {"name": "t", "description": "d", "parameters": {"type": "object"}},
        }
    ]


async def test_streams_text_and_usage() -> None:
    body = sse(
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        "[DONE]",
    )
    provider, seen = provider_with(body)
    chunks = [c async for c in provider.stream([Message("user", "hi")], [], "sys", 10)]

    assert chunks == [TextChunk("Hel"), TextChunk("lo"), UsageChunk(3, 2), DoneChunk()]
    assert seen[0]["stream"] is True
    assert "tools" not in seen[0]


async def test_maps_tool_call_fragments_by_index() -> None:
    body = sse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "add", "arguments": '{"a":'},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": " 1}"}}]}}]},
        "[DONE]",
    )
    provider, _ = provider_with(body)
    chunks = [c async for c in provider.stream([Message("user", "x")], [], "sys", 10)]

    assert chunks[:3] == [
        ToolCallStartChunk("call_1", "add"),
        ToolCallDeltaChunk("call_1", '{"a":'),
        ToolCallDeltaChunk("call_1", " 1}"),
    ]


async def test_http_error_raises_with_detail() -> None:
    provider, _ = provider_with(b'{"error": "bad key"}', status=401)
    with pytest.raises(ProviderError, match="401"):
        _ = [c async for c in provider.stream([Message("user", "x")], [], "sys", 10)]
