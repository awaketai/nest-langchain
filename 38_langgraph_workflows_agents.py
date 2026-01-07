# Workflows and agents
# 工作流具有预先设定的代码路径，并且被设计成按照特定顺序运行
# 代理动态决定流程和工具的使用方式

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

load_dotenv()

OPEN_API_URL = os.environ.get("OPEN_API_URL")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")

if OPEN_API_KEY is None:
    raise ValueError("OPENAI_KEY_V4 is not set")

llm = ChatOpenAI(
    base_url=OPEN_API_URL,
    api_key=SecretStr(OPEN_API_KEY),
    temperature=0,
)


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )


# Augment the LLM with schema for structured output
structured_llm = llm.with_structured_output(SearchQuery)

# Invoke the augmented LLM
output = structured_llm.invoke("How does Calcium CT store relate to high cholesterol?")


# Define a tool
def multiply(a: int, b: int) -> int:
    return a * b


# Augment the lLM with tools
llm_with_tools = llm.bind_tools([multiply])

# 执行调用
msg = llm_with_tools.invoke("What is 2 times 3?")
# 获取工具调用信息
msg.tool_calls
