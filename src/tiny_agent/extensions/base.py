"""Extension API. Subclass and override only the hooks you need."""

from __future__ import annotations

from dataclasses import dataclass

from ..types import Message, ToolCall, ToolLike, ToolResult


@dataclass(frozen=True)
class Block:
    """Return from ``before_tool_call`` to skip execution. ``reason`` is shown to the model."""

    reason: str = "Tool call was blocked by an extension."


class Extension:
    name: str = "extension"
    tools: tuple[ToolLike, ...] = ()

    async def before_send(self, messages: list[Message]) -> list[Message]:
        """Called before messages go to the model. May return a modified list."""
        return messages

    async def before_tool_call(self, tool_call: ToolCall) -> Block | None:
        """Return ``Block`` to stop the call. ``None`` lets it run."""
        return None

    async def after_tool_call(self, tool_call: ToolCall, result: ToolResult) -> ToolResult:
        """Called after a tool runs. May transform the result."""
        return result

    async def on_turn_done(self, messages: list[Message]) -> None:
        """Called when the model answered without requesting tools."""
