# LangChain的streaming system实现了实时更新
# Streaming

# 1.basic LLM Streaming

import asyncio
import os
from importlib.resources import Resource

from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain.agents.initialize import initialize_agent
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

OPEN_API_URL = os.environ.get("OPEN_API_URL")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")


if OPEN_API_KEY is None:
    raise ValueError("OPENAI_KEY_V4 is not set")
llm = ChatOpenAI(
    base_url=OPEN_API_URL,
    api_key=SecretStr(OPEN_API_KEY),
)


def basic_stream():
    response = llm.stream("Hello, world!")
    for chunk in response:
        print(chunk)

    print("-" * 30)


def chat_stream():
    for chunk in llm.stream("介绍一下人工智能"):
        print(chunk.content, end="", flush=True)

    print("-" * 30)


async def async_stream():
    """异步流式"""
    async for chunk in llm.astream("什么是机器学习"):
        print(chunk.content, end="", flush=True)

    print("-" * 30)


def chain_stream():
    prompt = ChatPromptTemplate.from_messages(
        [("system", "你是一个专业的技术顾问"), ("human", "{question}")]
    )
    output_parser = StrOutputParser()
    # 使用LCEL
    chain = prompt | llm | output_parser
    for chunk in chain.stream({"question": "解释一下什么是Docker"}):
        print(chunk, end="", flush=True)

    print("-" * 30)


def agent_stream():
    tools = load_tools(["llm-math"], llm=llm)
    # 创建agent
    agent = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    response = agent.invoke({"input": "25 的平方乘以 3 等于多少？"})
    print(response)
    print("-" * 30)


if __name__ == "__main__":
    # print("basic stream output:\n")
    # basic_stream()
    # print("chat stream output:\n")
    # chat_stream()
    # print("async chat stream output:\n")
    # asyncio.run(async_stream())
    # print("chain stream output:\n")
    # chain_stream()
    print("agent stream output:\n")
    agent_stream()
