"""Reflection agent demo built with LangChain.

The agent drafts an answer, critiques it, and revises when the critique finds
important gaps. This gives a compact example of the Reflection pattern.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
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


def build_drafter_agent(model: BaseChatModel):
    """Create the answer drafter agent with the local toolset."""
    return create_agent(
        model=model,
        tools=[calculator, current_time],
        system_prompt=(
            "You are the drafter in a Reflection workflow. "
            "Answer the user directly in Chinese unless asked otherwise. "
            "Use tools when they improve correctness."
        ),
    )


class ReflectionAgent:
    """A draft, critique, and revise agent."""

    def __init__(self, model: BaseChatModel | None = None, max_reflections: int = 2) -> None:
        self.model = model or build_model()
        self.drafter = build_drafter_agent(self.model)
        self.max_reflections = max_reflections

    def draft(self, question: str) -> str:
        """Create the first answer with tool access."""
        result = self.drafter.invoke({"messages": [{"role": "user", "content": question}]})
        return str(result["messages"][-1].content)

    def reflect(self, question: str, answer: str) -> dict[str, str | bool]:
        """Critique an answer and decide whether it needs revision."""
        response = self.model.invoke(
            [
                SystemMessage(
                    content=(
                        "你是 reflection critic。检查答案是否完整、准确、是否真正回答问题。"
                        "如果答案已经足够好，第一行输出 ACCEPT。"
                        "如果需要修改，第一行输出 REVISE，后面用简短中文列出必须修改的点。"
                        "不要直接重写答案。"
                    )
                ),
                HumanMessage(content=f"用户问题：{question}\n\n候选答案：{answer}"),
            ]
        )
        critique = str(response.content).strip()
        needs_revision = not critique.upper().startswith("ACCEPT")
        return {"needs_revision": needs_revision, "critique": critique}

    def revise(self, question: str, answer: str, critique: str) -> str:
        """Revise the answer using the critique, with tool access if needed."""
        prompt = (
            f"用户问题：{question}\n\n"
            f"当前答案：{answer}\n\n"
            f"反思意见：{critique}\n\n"
            "请根据反思意见修订答案。只输出修订后的最终答案。"
        )
        result = self.drafter.invoke({"messages": [{"role": "user", "content": prompt}]})
        return str(result["messages"][-1].content)

    def invoke(self, question: str) -> dict[str, Any]:
        """Run the full Reflection workflow."""
        answer = self.draft(question)
        reflections: list[dict[str, str | bool]] = []

        for _ in range(self.max_reflections):
            reflection = self.reflect(question, answer)
            reflections.append(reflection)
            if not reflection["needs_revision"]:
                break
            answer = self.revise(question, answer, str(reflection["critique"]))

        return {
            "question": question,
            "reflections": reflections,
            "answer": answer,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Reflection agent.")
    parser.add_argument(
        "question",
        nargs="?",
        default="现在 UTC 时间是多少？再计算 sqrt(144) + 3。",
        help="Question to ask the agent.",
    )
    parser.add_argument(
        "--max-reflections",
        type=int,
        default=2,
        help="Maximum critique-and-revise rounds.",
    )
    args = parser.parse_args()

    agent = ReflectionAgent(max_reflections=max(0, args.max_reflections))
    result = agent.invoke(args.question)

    print(f"问题：{result['question']}\n")
    print("反思过程：")
    for index, reflection in enumerate(result["reflections"], start=1):
        status = "需要修订" if reflection["needs_revision"] else "接受"
        print(f"{index}. {status}\n   {reflection['critique']}")

    print(f"\n最终答案：\n{result['answer']}")


if __name__ == "__main__":
    main()
