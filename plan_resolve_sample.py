"""Plan-and-resolve agent demo built with LangChain.

The agent first creates a short plan, then resolves each step with a tool-using
LangChain agent, and finally synthesizes the step results into one answer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool


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


def build_model() -> BaseChatModel:
    """Create the chat model from local environment variables."""
    load_dotenv()
    return ChatOpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL_ID"),
    )


def build_resolver_agent(model: BaseChatModel):
    """Create the step resolver agent with the local toolset."""
    return create_agent(
        model=model,
        tools=[calculator, current_time],
        system_prompt=(
            "You are the resolver in a Plan-and-Resolve workflow. "
            "Solve only the current step. Use tools when useful. "
            "Return a concise Chinese result with any important evidence."
        ),
    )


class PlanAndResolveAgent:
    """A simple planner, step resolver, and final answer synthesizer."""

    def __init__(self, model: BaseChatModel | None = None) -> None:
        self.model = model or build_model()
        self.resolver = build_resolver_agent(self.model)

    def plan(self, question: str) -> list[str]:
        """Ask the model for an executable plan and parse it into steps."""
        response = self.model.invoke(
            [
                SystemMessage(
                    content=(
                        "你是 planner。请把用户问题拆成 2 到 5 个可执行步骤。"
                        "只输出编号列表，每行一个步骤，不要执行步骤，不要给最终答案。"
                    )
                ),
                HumanMessage(content=question),
            ]
        )
        steps = parse_plan(str(response.content))
        if steps:
            return steps
        return [question]

    def resolve_step(
        self,
        question: str,
        step: str,
        previous_results: list[tuple[str, str]],
    ) -> str:
        """Resolve a single plan step with context from earlier steps."""
        context = format_previous_results(previous_results)
        prompt = (
            f"原始问题：{question}\n\n"
            f"之前步骤结果：\n{context}\n\n"
            f"当前步骤：{step}\n\n"
            "请只完成当前步骤。"
        )
        result = self.resolver.invoke({"messages": [{"role": "user", "content": prompt}]})
        return str(result["messages"][-1].content)

    def finalize(self, question: str, results: list[tuple[str, str]]) -> str:
        """Synthesize all resolved steps into the final answer."""
        response = self.model.invoke(
            [
                SystemMessage(
                    content=(
                        "你是 finalizer。根据已完成步骤结果回答原始问题。"
                        "不要编造没有出现在步骤结果中的事实。用中文清晰简洁回答。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"原始问题：{question}\n\n"
                        f"已完成步骤结果：\n{format_previous_results(results)}"
                    )
                ),
            ]
        )
        return str(response.content)

    def invoke(self, question: str) -> dict[str, Any]:
        """Run the full Plan-and-Resolve workflow."""
        steps = self.plan(question)
        results: list[tuple[str, str]] = []

        for step in steps:
            result = self.resolve_step(question, step, results)
            results.append((step, result))

        return {
            "question": question,
            "plan": steps,
            "steps": results,
            "answer": self.finalize(question, results),
        }


def parse_plan(text: str) -> list[str]:
    """Parse a numbered or bulleted plan into plain step strings."""
    steps: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)、])\s*", "", cleaned).strip()
        if cleaned:
            steps.append(cleaned)
    return steps


def format_previous_results(results: list[tuple[str, str]]) -> str:
    """Format completed step results for model context."""
    if not results:
        return "无"
    lines: list[str] = []
    for index, (step, result) in enumerate(results, start=1):
        lines.append(f"{index}. 步骤：{step}\n   结果：{result}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Plan-and-Resolve agent.")
    parser.add_argument(
        "question",
        nargs="?",
        default="现在 UTC 时间是多少？再计算 sqrt(144) + 3。",
        help="Question to ask the agent.",
    )
    args = parser.parse_args()

    agent = PlanAndResolveAgent()
    result = agent.invoke(args.question)

    print(f"问题：{result['question']}\n")
    print("计划：")
    for index, step in enumerate(result["plan"], start=1):
        print(f"{index}. {step}")

    print("\n执行结果：")
    for index, (step, step_result) in enumerate(result["steps"], start=1):
        print(f"{index}. {step}\n   {step_result}")

    print(f"\n最终答案：\n{result['answer']}")


if __name__ == "__main__":
    main()
