from __future__ import annotations

from collections.abc import AsyncIterator

from tiny_agent import Agent, AgentEvent, Message, ToolDefinition
from tiny_agent.types import (
    DoneChunk,
    ProviderChunk,
    TextChunk,
    ToolCallDeltaChunk,
    ToolCallStartChunk,
)


class FakeProvider:
    """Replays scripted turns, one list of chunks per model call. No network."""

    name = "fake"

    def __init__(self, turns: list[list[ProviderChunk]]) -> None:
        self._turns = list(turns)
        self.calls: list[list[Message]] = []

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        system_prompt: str,
        max_tokens: int,
    ) -> AsyncIterator[ProviderChunk]:
        self.calls.append(list(messages))
        for chunk in self._turns.pop(0):
            yield chunk
        yield DoneChunk()


def text_turn(text: str) -> list[ProviderChunk]:
    return [TextChunk(text)]


def tool_turn(call_id: str, name: str, args_json: str) -> list[ProviderChunk]:
    # Split the arguments to exercise fragment accumulation.
    mid = len(args_json) // 2
    return [
        ToolCallStartChunk(call_id, name),
        ToolCallDeltaChunk(call_id, args_json[:mid]),
        ToolCallDeltaChunk(call_id, args_json[mid:]),
    ]


async def collect(agent: Agent, prompt: str) -> list[AgentEvent]:
    return [event async for event in agent.run(prompt)]
