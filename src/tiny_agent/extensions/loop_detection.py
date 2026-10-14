"""Break infinite loops of identical tool calls."""

from __future__ import annotations

import json

from ..types import ToolCall
from .base import Block, Extension


class LoopDetection(Extension):
    """Blocks a tool call when the last ``threshold`` calls were identical.

    The block reason tells the model to stop and explain, which usually breaks the loop.
    """

    name = "loop-detection"

    def __init__(self, threshold: int = 3) -> None:
        if threshold < 2:
            raise ValueError("threshold must be at least 2")
        self._threshold = threshold
        self._history: list[str] = []

    async def before_tool_call(self, tool_call: ToolCall) -> Block | None:
        signature = f"{tool_call.name}:{json.dumps(tool_call.arguments, sort_keys=True)}"
        self._history.append(signature)
        recent = self._history[-self._threshold :]
        if len(recent) == self._threshold and all(s == recent[0] for s in recent):
            self._history.clear()
            return Block(
                "Loop detected: you are repeating the same tool calls. "
                "Stop and explain what you are trying to do differently."
            )
        return None
