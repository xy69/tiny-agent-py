"""Tiny CLI chat with one tool. Works with any OpenAI-compatible endpoint.

    # OpenRouter
    OPENAI_API_KEY=sk-... uv run python examples/cli.py --base-url https://openrouter.ai/api/v1 \
        --model openai/gpt-4o-mini

    # Local model via Ollama (no key needed)
    uv run python examples/cli.py --base-url http://localhost:11434/v1 --model qwen2.5:7b
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime

from pydantic import BaseModel

from tiny_agent import Agent, LoopDetection, TextDelta, ToolCallFinished, TurnDone, tool
from tiny_agent.providers import OpenAICompatibleProvider


class NoArgs(BaseModel):
    pass


@tool("get_time", "Get the current UTC date and time in ISO 8601", NoArgs)
async def get_time(_: NoArgs) -> str:
    return datetime.now(UTC).isoformat()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    provider = OpenAICompatibleProvider(
        args.model, api_key=os.environ.get("OPENAI_API_KEY"), base_url=args.base_url
    )
    agent = Agent(
        provider,
        "You are a concise assistant. Use tools when they help.",
        tools=[get_time],
        extensions=[LoopDetection()],
    )

    try:
        while True:
            prompt = await asyncio.to_thread(input, "\n> ")
            async for event in agent.run(prompt):
                if isinstance(event, TextDelta):
                    print(event.text, end="", flush=True)
                elif isinstance(event, ToolCallFinished):
                    print(f"\n[tool {event.tool_call.name} -> {event.result.output}]")
                elif isinstance(event, TurnDone):
                    print()
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
