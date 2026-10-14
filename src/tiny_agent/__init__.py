"""tiny-agent: the loop and the types. Everything else is an extension."""

from .agent import Agent
from .extensions import Block, Extension, LoopDetection
from .tools import Tool, tool
from .types import (
    AgentEvent,
    ErrorEvent,
    Message,
    MessageStore,
    Provider,
    TextDelta,
    ToolCall,
    ToolCallFinished,
    ToolCallStarted,
    ToolDefinition,
    ToolResult,
    TurnDone,
    Usage,
)

__all__ = [
    "Agent",
    "AgentEvent",
    "Block",
    "ErrorEvent",
    "Extension",
    "LoopDetection",
    "Message",
    "MessageStore",
    "Provider",
    "TextDelta",
    "Tool",
    "ToolCall",
    "ToolCallFinished",
    "ToolCallStarted",
    "ToolDefinition",
    "ToolResult",
    "TurnDone",
    "Usage",
    "tool",
]
