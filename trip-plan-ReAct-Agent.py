from __future__ import annotations
import os
from typing import Any
from urllib.parse import urlencode

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_openai import ChatOpenAI
import requests

@tool
def search_keywords_tool(keywords: str = None, types: str = None, topk: int = 3, **kwargs) -> tuple[list, list]:
    """根据关键词或高德 POI 类型码搜索地点，返回匹配地点的地址和经纬度。

    适用于查询景点、餐厅、酒店、交通站点等地点信息。keywords 和 types
    至少传入一个；如果同时传入，会按关键词和类型共同筛选。可通过 kwargs
    透传高德地图 Place Text Search API 支持的其他参数，例如 city、city_limit、
    show_fields、page_size、page_num、sig。

    Args:
        keywords: 地点关键词，例如“灵隐寺”“西湖”“杭州东站”。
        types: 高德 POI 类型码，例如“110000”表示风景名胜相关类型。
        topk: 返回前几个搜索结果，默认返回 3 个。
        **kwargs: 其他高德地图检索参数。

    Returns:
        一个二元组：(address, locations)，address 为地址列表，
        locations 为对应的经纬度字符串列表，格式通常为 "经度,纬度"。
    """
    key = os.getenv("ALI_API_KEY")
    base_url = "https://restapi.amap.com/v5/place/text"
    params = {
        "key": key,
        "keywords": keywords,
        "types": types,
        **kwargs
    }

    if not keywords and not types:
        raise ValueError("keyword 或者 types 二选一必填")

    s = urlencode({k: v for k, v in params.items() if v})
    url = f"{base_url}?{s}"
    response = requests.get(url).json()
    locations_list = response["pois"]

    address = []
    locations = []
    for i in range(topk):
        address.append(locations_list[i]["address"])
        locations.append(locations_list[i]["location"])

    return address, locations

def plan_route_tool(origin: str, destination: str, **kwargs) -> list:
    """规划两点之间的步行路线，返回分步骤的导航指令。

    适用于根据起点和终点经纬度生成步行导航方案，例如从酒店到景点、
    从景点到餐厅、从地铁站到目的地等短距离路线规划。origin 和
    destination 均使用 "经度,纬度" 格式。可通过 kwargs 透传高德地图
    Walking Direction API 支持的其他参数。

    Args:
        origin: 起点经纬度，格式为 "经度,纬度"，例如 "116.434307,39.90909"。
        destination: 终点经纬度，格式为 "经度,纬度"。
        **kwargs: 其他高德地图步行路线规划参数。

    Returns:
        步行路线的导航指令列表，每一项是一段路线说明。
    """
    key = os.getenv("ALI_API_KEY")
    base_url = "https://restapi.amap.com/v5/direction/walking"
    params = {
        "key": key,
        "origin": origin,
        "destination": destination,
        **kwargs
    }
    s = urlencode({k: v for k, v in params.items() if v})
    url = f"{base_url}?{s}"
    # print(url)
    response = requests.get(url).json()
    paths = []
    for chunk in response["route"]["paths"][0]["steps"]:
        paths.append(chunk["instruction"])
    
    return paths

def build_model() -> BaseChatModel:
    """Create the chat model from local environment variables."""
    load_dotenv()
    return ChatOpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL_ID"),
    )

def build_agent(model: BaseChatModel):
    """Create the step resolver agent with the local toolset."""
    return create_agent(
        model=model,
        tools=[plan_route_tool, search_keywords_tool],
        system_prompt=(
            "You are a concise ReAct-style assistant. Use tools when they help. "
            "Explain the final answer clearly in Chinese unless the user asks otherwise."
        ),
    )
if __name__ == "__main__":
    llm = build_model()
    agent = build_agent(llm)
    response = agent.invoke({"messages": [{"role": "user", "content": "我现在在西湖，我想去灵隐寺，请问路怎么走？"}]})

    print(response["messages"][-1].content)