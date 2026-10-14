"""Core types. Plain dataclasses and Protocols: no framework required."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass(frozen=True)
class ToolResult:
    output: str
    is_error: bool = False


class ToolLike(Protocol):
    """Anything the agent can execute."""

    @property
    def definition(self) -> ToolDefinition: ...

    async def execute(self, arguments: dict[str, Any]) -> ToolResult: ...


# --- Provider stream (what a model backend emits) -------------------------------------------


@dataclass(frozen=True)
class TextChunk:
    text: str


@dataclass(frozen=True)
class ToolCallStartChunk:
    id: str
    name: str


@dataclass(frozen=True)
class ToolCallDeltaChunk:
    id: str
    arguments: str  # a fragment of the JSON arguments string


@dataclass(frozen=True)
class UsageChunk:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class DoneChunk:
    pass


ProviderChunk = TextChunk | ToolCallStartChunk | ToolCallDeltaChunk | UsageChunk | DoneChunk


class Provider(Protocol):
    name: str

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        system_prompt: str,
        max_tokens: int,
    ) -> AsyncIterator[ProviderChunk]: ...


# --- Agent events (what a caller of Agent.run sees) -----------------------------------------


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallStarted:
    tool_call: ToolCall


@dataclass(frozen=True)
class ToolCallFinished:
    tool_call: ToolCall
    result: ToolResult


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ErrorEvent:
    message: str


@dataclass(frozen=True)
class TurnDone:
    pass


AgentEvent = TextDelta | ToolCallStarted | ToolCallFinished | Usage | ErrorEvent | TurnDone


# --- Persistence ----------------------------------------------------------------------------


class MessageStore(Protocol):
    def get_messages(self) -> list[Message]: ...

    def append(self, message: Message) -> None: ...
