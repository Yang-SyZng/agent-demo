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


def build_model() -> BaseChatModel: