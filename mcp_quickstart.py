"""
LangChain MCP Adapters 快速入门示例

这是一个最简单的示例，展示如何使用 langchain-mcp-adapters 调用 MCP 服务
"""

import asyncio
import os

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import create_mcp_client
from langchain_openai import ChatOpenAI


async def quickstart():
    """快速入门示例"""

    # 1. 配置 API 密钥（请替换为你的实际 API 密钥）
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"

    # 如果使用 DeepSeek
    # os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"
    # os.environ["OPENAI_API_KEY"] = "your-deepseek-api-key"

    print("🚀 LangChain MCP 快速入门\n")

    # 2. 连接到 MCP 服务器
    print("📡 正在连接到 MCP 服务器...")
    async with create_mcp_client(
        command="python",
        args=["mcp_server_example.py"],
    ) as client:
        # 3. 获取所有可用工具
        tools = await client.list_tools()
        print(f"✅ 成功加载 {len(tools)} 个工具\n")

        # 显示工具列表
        print("📋 可用工具:")
        for i, tool in enumerate(tools, 1):
            print(f"   {i}. {tool.name}: {tool.description}")

        # 4. 创建 LLM
        print("\n🤖 初始化 LLM...")
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

        # 5. 创建 Agent 提示模板
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个有用的助手，可以使用工具来帮助用户。"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # 6. 创建 Agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
        )

        # 7. 测试工具调用
        print("\n" + "=" * 60)
        print("开始测试")
        print("=" * 60 + "\n")

        # 测试问题列表
        test_questions = [
            "北京的天气怎么样？",
            "计算 25 乘以 48",
            "查询数据库中的用户信息",
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"【问题 {i}】{question}")
            response = await agent_executor.ainvoke({"input": question})
            print(f"【回答】{response['output']}\n")
            print("-" * 60 + "\n")

        print("✨ 测试完成！")


async def simple_tool_call():
    """简单的工具调用示例（不使用 Agent）"""

    print("🔧 直接调用工具示例\n")

    async with create_mcp_client(
        command="python",
        args=["mcp_server_example.py"],
    ) as client:
        # 获取工具
        tools = await client.list_tools()

        # 查找天气工具
        weather_tool = next((t for t in tools if t.name == "get_weather"), None)

        if weather_tool:
            # 直接调用工具
            print("调用工具: get_weather")
            result = await weather_tool.ainvoke({"city": "上海"})
            print(f"结果:\n{result}\n")

        # 查找计算器工具
        calc_tool = next((t for t in tools if t.name == "calculator"), None)

        if calc_tool:
            print("调用工具: calculator")
            result = await calc_tool.ainvoke(
                {"operation": "multiply", "a": 12, "b": 34}
            )
            print(f"结果:\n{result}\n")


async def interactive_mode():
    """交互式模式"""

    print("💬 交互式模式（输入 'quit' 或 'exit' 退出）\n")

    async with create_mcp_client(
        command="python",
        args=["mcp_server_example.py"],
    ) as client:
        tools = await client.list_tools()
        print(f"✅ 加载了 {len(tools)} 个工具\n")

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个有用的助手，可以使用工具来帮助用户。"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

        # 交互循环
        while True:
            try:
                user_input = input("👤 你: ").strip()

                if user_input.lower() in ["quit", "exit", "退出"]:
                    print("👋 再见！")
                    break

                if not user_input:
                    continue

                response = await agent_executor.ainvoke({"input": user_input})
                print(f"🤖 助手: {response['output']}\n")

            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {e}\n")


def main():
    """主函数"""

    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║  LangChain MCP Adapters - 快速入门                       ║
║                                                          ║
║  选择运行模式:                                           ║
║  1. 快速入门示例（推荐）                                 ║
║  2. 简单工具调用                                         ║
║  3. 交互式模式                                           ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)

    # 取消注释下面的代码来运行不同的示例

    # 运行快速入门示例（推荐）
    asyncio.run(quickstart())

    # 或运行简单工具调用
    # asyncio.run(simple_tool_call())

    # 或运行交互式模式
    # asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
