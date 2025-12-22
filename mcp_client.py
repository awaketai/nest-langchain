"""
LangChain 使用 langchain-mcp-adapters 调用 MCP 服务的示例
修复版本
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import create_mcp_client
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

OPEN_API_URL = os.environ.get("OPEN_API_URL")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")

if OPEN_API_KEY is None:
    raise ValueError("OPEN_API_KEY is not set")


async def main():
    """主函数"""

    print("=" * 60)
    print("LangChain MCP 适配器示例")
    print("=" * 60)

    # 1. 连接到 MCP 服务器
    print("\n📡 连接到 MCP 服务器...")

    async with create_mcp_client(
        command="python",
        args=["34_model_context_mcp_server.py"],
    ) as client:
        print("✅ 成功连接到 MCP 服务器")

        # 2. 获取所有可用的工具
        print("\n🔧 加载 MCP 工具...")
        all_tools = await client.list_tools()
        print(f"✅ 成功加载 {len(all_tools)} 个工具:")
        for tool in all_tools:
            print(f"   - {tool.name}: {tool.description}")

        # 3. 创建 LLM
        print("\n🤖 初始化 LLM...")
        model = ChatOpenAI(
            base_url=OPEN_API_URL,
            api_key=SecretStr(OPEN_API_KEY),
            model="gpt-4o-mini",
            temperature=0,
        )

        # 4. 创建 Agent 提示模板
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个有用的助手，可以使用各种工具来帮助用户。"
                    "请使用提供的工具来回答用户的问题。",
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # 5. 创建 Agent
        print("🎯 创建 Agent...")
        agent = create_tool_calling_agent(model, all_tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=all_tools,
            verbose=True,
            handle_parsing_errors=True,
        )

        # 6. 测试各种工具调用
        print("\n" + "=" * 60)
        print("开始测试工具调用")
        print("=" * 60)

        # 测试 1: 天气查询
        print("\n【测试 1: 天气查询】")
        response = await agent_executor.ainvoke(
            {"input": "北京和上海的天气怎么样？", "chat_history": []}
        )
        print(f"\n回答: {response['output']}")

        # 测试 2: 计算器
        print("\n" + "-" * 60)
        print("【测试 2: 数学计算】")
        response = await agent_executor.ainvoke(
            {"input": "计算 123 乘以 456 等于多少？", "chat_history": []}
        )
        print(f"\n回答: {response['output']}")

        # 测试 3: 数据库查询
        print("\n" + "-" * 60)
        print("【测试 3: 数据库查询】")
        response = await agent_executor.ainvoke(
            {"input": "查询数据库中所有的用户信息", "chat_history": []}
        )
        print(f"\n回答: {response['output']}")

        # 测试 4: 文本分析
        print("\n" + "-" * 60)
        print("【测试 4: 文本分析】")
        response = await agent_executor.ainvoke(
            {
                "input": "分析这段文本：'LangChain 是一个强大的框架，用于构建 LLM 应用程序。"
                "它提供了丰富的工具和集成。'",
                "chat_history": [],
            }
        )
        print(f"\n回答: {response['output']}")

        # 测试 5: 复杂查询（需要多个工具）
        print("\n" + "-" * 60)
        print("【测试 5: 复杂查询】")
        response = await agent_executor.ainvoke(
            {
                "input": "查询产品表中价格大于3000的产品，然后告诉我北京的天气",
                "chat_history": [],
            }
        )
        print(f"\n回答: {response['output']}")

        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
