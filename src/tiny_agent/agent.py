"""The agent loop: stream model, parse tool calls, run hooks, execute tools, repeat."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from .extensions.base import Block, Extension
from .types import (
    AgentEvent,
    ErrorEvent,
    Message,
    MessageStore,
    Provider,
    TextChunk,
    TextDelta,
    ToolCall,
    ToolCallDeltaChunk,
    ToolCallFinished,
    ToolCallStartChunk,
    ToolCallStarted,
    ToolDefinition,
    ToolLike,
    ToolResult,
    TurnDone,
    Usage,
    UsageChunk,
)


@dataclass
class _PendingToolCall:
    name: str
    arguments: str = ""


def _parse_arguments(raw: str) -> dict[str, Any]:
    """Parse streamed JSON arguments. Unparseable input is kept under ``_raw`` so the tool's own
    validation can reject it and the model gets a chance to retry."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_raw": raw}


class Agent:
    def __init__(
        self,
        provider: Provider,
        system_prompt: str,
        *,
        max_tokens: int = 4096,
        max_steps: int = 50,
        tools: Sequence[ToolLike] = (),
        extensions: Sequence[Extension] = (),
        store: MessageStore | None = None,
    ) -> None:
        self._provider = provider
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._max_steps = max_steps
        self._extensions = list(extensions)
        self._store = store
        self._tools: dict[str, ToolLike] = {}
        for t in [*tools, *(t for ext in self._extensions for t in ext.tools)]:
            self._tools[t.definition.name] = t
        self._messages: list[Message] = store.get_messages() if store else []

    @property
    def messages(self) -> list[Message]:
        return self._messages

    def _append(self, message: Message) -> None:
        self._messages.append(message)
        if self._store:
            self._store.append(message)

    def _tool_definitions(self) -> list[ToolDefinition]:
        return [t.definition for t in self._tools.values()]

    async def run(self, user_message: str) -> AsyncIterator[AgentEvent]:
        self._append(Message("user", user_message))

        for _ in range(self._max_steps):
            to_send = list(self._messages)
            for ext in self._extensions:
                to_send = await ext.before_send(to_send)

            pending: dict[str, _PendingToolCall] = {}
            text = ""

            async for chunk in self._provider.stream(
                to_send, self._tool_definitions(), self._system_prompt, self._max_tokens
            ):
                if isinstance(chunk, TextChunk):
                    text += chunk.text
                    yield TextDelta(chunk.text)
                elif isinstance(chunk, ToolCallStartChunk):
                    pending[chunk.id] = _PendingToolCall(chunk.name)
                elif isinstance(chunk, ToolCallDeltaChunk):
                    target = pending.get(chunk.id) or (
                        list(pending.values())[-1] if pending else None
                    )
                    if target:
                        target.arguments += chunk.arguments
                elif isinstance(chunk, UsageChunk):
                    yield Usage(chunk.input_tokens, chunk.output_tokens)

            calls = [ToolCall(i, p.name, _parse_arguments(p.arguments)) for i, p in pending.items()]
            self._append(Message("assistant", text, tool_calls=calls))

            if not calls:
                for ext in self._extensions:
                    await ext.on_turn_done(self._messages)
                yield TurnDone()
                return

            for call in calls:
                async for event in self._run_tool_call(call):
                    yield event

        yield ErrorEvent(f"Stopped after {self._max_steps} steps without a final answer.")

    async def _run_tool_call(self, call: ToolCall) -> AsyncIterator[AgentEvent]:
        for ext in self._extensions:
            decision = await ext.before_tool_call(call)
            if isinstance(decision, Block):
                # A tool result is still required to keep the message sequence valid.
                result = ToolResult(decision.reason, is_error=True)
                self._append(Message("tool", result.output, tool_call_id=call.id))
                yield ToolCallFinished(call, result)
                return

        yield ToolCallStarted(call)

        tool = self._tools.get(call.name)
        if tool is None:
            result = ToolResult(f"Unknown tool: {call.name}", is_error=True)
        else:
            try:
                result = await tool.execute(call.arguments)
            except Exception as err:  # tool failures are data for the model, not crashes
                result = ToolResult(f"Tool error: {err}", is_error=True)

        for ext in self._extensions:
            result = await ext.after_tool_call(call, result)

        yield ToolCallFinished(call, result)
        self._append(Message("tool", result.output, tool_call_id=call.id))
