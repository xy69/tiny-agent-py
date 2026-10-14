"""Provider for any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI, OpenRouter, Ollama,
llama.cpp server, ...), implemented directly over httpx so the SSE handling is visible."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..types import (
    DoneChunk,
    Message,
    ProviderChunk,
    TextChunk,
    ToolCallDeltaChunk,
    ToolCallStartChunk,
    ToolDefinition,
    UsageChunk,
)


class ProviderError(RuntimeError):
    pass


def format_messages(messages: list[Message], system_prompt: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in m.tool_calls
                ]
            out.append(entry)
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        else:
            out.append({"role": "user", "content": m.content})
    return out


def format_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def parse_sse_data(lines: list[str]) -> list[str]:
    """Extract ``data:`` payloads from raw SSE lines, ignoring comments and keep-alives."""
    return [ln[5:].strip() for ln in lines if ln.startswith("data:")]


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        system_prompt: str,
        max_tokens: int,
    ) -> AsyncIterator[ProviderChunk]:
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": format_messages(messages, system_prompt),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = format_tools(tools)

        # Providers stream tool calls by index: only the first fragment carries id and name.
        index_to_id: dict[int, str] = {}

        async with self._client.stream(
            "POST", self._url, json=body, headers=self._headers
        ) as response:
            if response.status_code >= 400:
                detail = (await response.aread()).decode(errors="replace")[:500]
                raise ProviderError(f"HTTP {response.status_code}: {detail}")

            async for line in response.aiter_lines():
                payloads = parse_sse_data([line])
                if not payloads:
                    continue
                data = payloads[0]
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = event.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}

                if content := delta.get("content"):
                    yield TextChunk(content)

                for tc in delta.get("tool_calls") or []:
                    index = tc.get("index", 0)
                    fn = tc.get("function") or {}
                    if tc.get("id") and fn.get("name"):
                        index_to_id[index] = tc["id"]
                        yield ToolCallStartChunk(tc["id"], fn["name"])
                    if fragment := fn.get("arguments"):
                        yield ToolCallDeltaChunk(index_to_id.get(index, f"tool_{index}"), fragment)

                if usage := event.get("usage"):
                    yield UsageChunk(
                        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
                    )

        yield DoneChunk()
