from __future__ import annotations

from pydantic import BaseModel, Field

from tiny_agent import (
    Agent,
    Block,
    ErrorEvent,
    Extension,
    LoopDetection,
    TextDelta,
    ToolCall,
    ToolCallFinished,
    ToolCallStarted,
    TurnDone,
    tool,
)

from .helpers import FakeProvider, collect, text_turn, tool_turn


class AddArgs(BaseModel):
    a: int
    b: int = Field(description="second addend")


@tool("add", "Add two integers", AddArgs)
async def add(args: AddArgs) -> str:
    return str(args.a + args.b)


async def test_plain_answer_ends_turn() -> None:
    agent = Agent(FakeProvider([text_turn("hello")]), "sys")
    events = await collect(agent, "hi")
    assert events == [TextDelta("hello"), TurnDone()]
    assert [m.role for m in agent.messages] == ["user", "assistant"]


async def test_tool_call_round_trip() -> None:
    provider = FakeProvider([tool_turn("c1", "add", '{"a": 2, "b": 3}'), text_turn("5")])
    agent = Agent(provider, "sys", tools=[add])
    events = await collect(agent, "2+3?")

    finished = [e for e in events if isinstance(e, ToolCallFinished)]
    assert [e.result.output for e in finished] == ["5"]
    assert isinstance(events[-1], TurnDone)
    # Second model call must see the tool result.
    assert provider.calls[1][-1].role == "tool"
    assert provider.calls[1][-1].tool_call_id == "c1"


async def test_invalid_arguments_are_returned_to_the_model() -> None:
    provider = FakeProvider([tool_turn("c1", "add", '{"a": "x"}'), text_turn("sorry")])
    agent = Agent(provider, "sys", tools=[add])
    events = await collect(agent, "go")

    result = next(e for e in events if isinstance(e, ToolCallFinished)).result
    assert result.is_error
    assert "a:" in result.output and "b:" in result.output


async def test_unparseable_json_arguments_become_a_validation_error() -> None:
    provider = FakeProvider([tool_turn("c1", "add", '{"a": 1, "b"'), text_turn("ok")])
    agent = Agent(provider, "sys", tools=[add])
    events = await collect(agent, "go")
    assert next(e for e in events if isinstance(e, ToolCallFinished)).result.is_error


async def test_unknown_tool_is_an_error_result_not_a_crash() -> None:
    provider = FakeProvider([tool_turn("c1", "nope", "{}"), text_turn("ok")])
    events = await collect(Agent(provider, "sys"), "go")
    result = next(e for e in events if isinstance(e, ToolCallFinished)).result
    assert result.is_error and "Unknown tool" in result.output


async def test_tool_exception_is_an_error_result() -> None:
    class Empty(BaseModel):
        pass

    @tool("boom", "always fails", Empty)
    async def boom(_: Empty) -> str:
        raise RuntimeError("kaput")

    provider = FakeProvider([tool_turn("c1", "boom", "{}"), text_turn("ok")])
    events = await collect(Agent(provider, "sys", tools=[boom]), "go")
    result = next(e for e in events if isinstance(e, ToolCallFinished)).result
    assert result.is_error and "kaput" in result.output


async def test_before_tool_call_block_skips_execution_but_keeps_sequence_valid() -> None:
    class Deny(Extension):
        async def before_tool_call(self, tool_call: ToolCall) -> Block | None:
            return Block("not allowed")

    provider = FakeProvider([tool_turn("c1", "add", '{"a": 1, "b": 1}'), text_turn("ok")])
    agent = Agent(provider, "sys", tools=[add], extensions=[Deny()])
    events = await collect(agent, "go")

    assert not any(isinstance(e, ToolCallStarted) for e in events)
    result = next(e for e in events if isinstance(e, ToolCallFinished)).result
    assert result.is_error and result.output == "not allowed"
    assert provider.calls[1][-1].role == "tool"


async def test_after_tool_call_can_transform_result() -> None:
    from tiny_agent import ToolResult

    class Shout(Extension):
        async def after_tool_call(self, tool_call: ToolCall, result: ToolResult) -> ToolResult:
            return ToolResult(result.output + "!", result.is_error)

    provider = FakeProvider([tool_turn("c1", "add", '{"a": 1, "b": 1}'), text_turn("ok")])
    events = await collect(Agent(provider, "sys", tools=[add], extensions=[Shout()]), "go")
    assert next(e for e in events if isinstance(e, ToolCallFinished)).result.output == "2!"


async def test_extension_tools_are_registered() -> None:
    class WithTool(Extension):
        tools = (add,)

    provider = FakeProvider([tool_turn("c1", "add", '{"a": 4, "b": 4}'), text_turn("ok")])
    events = await collect(Agent(provider, "sys", extensions=[WithTool()]), "go")
    assert next(e for e in events if isinstance(e, ToolCallFinished)).result.output == "8"


async def test_loop_detection_blocks_the_third_identical_call() -> None:
    same = tool_turn
    provider = FakeProvider(
        [
            same("c1", "add", '{"a": 1, "b": 1}'),
            same("c2", "add", '{"a": 1, "b": 1}'),
            same("c3", "add", '{"a": 1, "b": 1}'),
            text_turn("done"),
        ]
    )
    agent = Agent(provider, "sys", tools=[add], extensions=[LoopDetection(threshold=3)])
    events = await collect(agent, "go")

    results = [e.result for e in events if isinstance(e, ToolCallFinished)]
    assert [r.is_error for r in results] == [False, False, True]
    assert "Loop detected" in results[2].output


async def test_max_steps_stops_a_runaway_loop() -> None:
    provider = FakeProvider([tool_turn(f"c{i}", "add", '{"a": 1, "b": 1}') for i in range(5)])
    agent = Agent(provider, "sys", tools=[add], max_steps=3)
    events = await collect(agent, "go")
    assert isinstance(events[-1], ErrorEvent)
    assert len(provider.calls) == 3
