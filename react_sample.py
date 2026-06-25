"""Minimal LangChain ReAct-style agent demo.

The agent uses LangChain's agent loop: the model reasons, calls tools when
useful, observes results, and continues until it can answer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_openai import ChatOpenAI

@tool
def calculator(expression: str) -> str:
    """Evaluate a simple math expression using Python's math functions."""
    allowed_names: dict[str, Any] = {
        name: getattr(math, name) for name in dir(math) if not name.startswith("_")
    }
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
    except Exception as exc:  # noqa: BLE001 - tool should return failures to the model
        return f"Calculation error: {exc}"
    return str(result)


@tool
def current_time(timezone: str = "UTC") -> str:
    """Return the current time. Supports UTC and local."""
    now = dt.datetime.now(dt.UTC)
    if timezone.lower() == "local":
        now = dt.datetime.now().astimezone()
    return now.isoformat(timespec="seconds")


def build_agent():
    """Create a LangChain agent with a small local toolset."""
    load_dotenv()

    model = ChatOpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL_ID")
    )

    return create_agent(
        model=model,
        tools=[calculator, current_time],
        system_prompt=(
            "You are a concise ReAct-style assistant. Use tools when they help. "
            "Explain the final answer clearly in Chinese unless the user asks otherwise."
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a basic LangChain ReAct agent.")
    parser.add_argument(
        "question",
        nargs="?",
        default="现在 UTC 时间是多少？再计算 sqrt(144) + 3。",
        help="Question to ask the agent.",
    )
    args = parser.parse_args()

    agent = build_agent()
    print(args.question)
    result = agent.invoke({"messages": [{"role": "user", "content": args.question}]})
    final_message = result["messages"][-1]
    print(result)


if __name__ == "__main__":
    main()
    # build_agent()