"""Define tools from pydantic models: the schema sent to the model is the validator you run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from .types import ToolDefinition, ToolResult

InputT = TypeVar("InputT", bound=BaseModel)


class Tool(Generic[InputT]):
    """A tool whose JSON Schema and argument validation both come from one pydantic model.

    Invalid arguments are returned to the model as an error result instead of raising, so it can
    correct itself on the next step.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[InputT],
        handler: Callable[[InputT], Awaitable[str]],
    ) -> None:
        self._input_model = input_model
        self._handler = handler
        self._definition = ToolDefinition(
            name=name,
            description=description,
            parameters=input_model.model_json_schema(),
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            parsed = self._input_model.model_validate(arguments)
        except ValidationError as err:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}"
                for e in err.errors(include_url=False)
            )
            return ToolResult(f"Invalid arguments: {problems}", is_error=True)
        return ToolResult(await self._handler(parsed))


def tool(
    name: str, description: str, input_model: type[InputT]
) -> Callable[[Callable[[InputT], Awaitable[str]]], Tool[InputT]]:
    """Decorator form of :class:`Tool`."""

    def wrap(handler: Callable[[InputT], Awaitable[str]]) -> Tool[InputT]:
        return Tool(name, description, input_model, handler)

    return wrap
