# tiny-agent-py

Tiny, extensible AI agent for Python. The core is the loop and the types. Everything else is an extension.

A Python port of [`tiny-agent`](https://github.com/xy69/tiny-agent) (TypeScript), with two changes that suit Python: tools are defined from [pydantic](https://docs.pydantic.dev) models, and the model backend is plain `httpx` so the streaming code is easy to read.

## Install

```bash
uv add git+https://github.com/xy69/tiny-agent-py
```

Requires Python 3.11+.

## Usage

```python
import asyncio
from pydantic import BaseModel
from tiny_agent import Agent, LoopDetection, TextDelta, tool
from tiny_agent.providers import OpenAICompatibleProvider


class Weather(BaseModel):
    city: str


@tool("get_weather", "Get the current weather for a city", Weather)
async def get_weather(args: Weather) -> str:
    return f"Sunny in {args.city}"


async def main() -> None:
    provider = OpenAICompatibleProvider(
        "openai/gpt-4o-mini",
        api_key="...",
        base_url="https://openrouter.ai/api/v1",  # or http://localhost:11434/v1 for Ollama
    )
    agent = Agent(provider, "You are concise.", tools=[get_weather], extensions=[LoopDetection()])

    async for event in agent.run("What's the weather in Palma?"):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)

    await provider.aclose()


asyncio.run(main())
```

Try the interactive example against a hosted or local model:

```bash
uv run python examples/cli.py --base-url http://localhost:11434/v1 --model qwen2.5:7b
```

## How it works

1. Append the user message and call each extension's `before_send`.
2. Stream the model's response, collecting text and tool-call fragments.
3. No tool calls: call `on_turn_done`, emit `TurnDone`, return.
4. For each tool call: `before_tool_call` (may `Block`), execute, `after_tool_call`.
5. Append tool results and go back to step 2. `max_steps` (default 50) stops runaway loops.

Design choices:

- **Tool errors are data, not crashes.** Invalid arguments, unknown tools and exceptions are returned to the model as error results so it can correct itself. Argument JSON that cannot be parsed is passed to validation, which rejects it the same way.
- **Blocked calls still produce a tool result**, which keeps the message sequence valid for the provider API.
- **One definition per tool.** The pydantic model produces both the JSON Schema sent to the model and the validation applied to what comes back.

## Extensions

Subclass `Extension` and override only the hooks you need: `before_send`, `before_tool_call`, `after_tool_call`, `on_turn_done`. An extension can also contribute `tools`. Extensions run in list order, so put safety checks before others.

Included: `LoopDetection` (blocks repeated identical tool calls).

## Providers

Any object with `name` and an async-generator `stream(messages, tools, system_prompt, max_tokens)` works. `OpenAICompatibleProvider` covers OpenAI, OpenRouter, Ollama and llama.cpp's server.

## Persistence

Pass `store=` with `get_messages()` and `append()` to persist conversations.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
```

Tests use a scripted fake provider and an `httpx.MockTransport`, so they need no network or API key.

## Status

Early. Not yet ported from the TypeScript version: Anthropic provider, context compaction, sub-agent tasks, built-in file/shell tools, JSONL sessions.

## License

MIT
