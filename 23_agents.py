# 在LangChain中代理(Agent)赋予模型推理和行动的能力，与链(Chain)预定义了一系列固定的步骤不同
# 代理能够根据当前情况和可用的工具，动态的巨鼎下一步应该做什么
#
# 什么是代理
# 代理的核心思想是：LLM + 工具 + 循环
# 1.LLM作为大脑，代理使用一个LLM作为其推理引擎，LLM负责分析用户请求、思考如何解决问题、决定使用哪个工具以及如何使用它
# 2.工具作为手臂：代理被赋予一套工具(Tools)，这些工具时LLM可以调用的函数，用于执行特定任务，例如
# 搜索互联网、执行代码、查询数据库、调用API、进行数学计算
# 3.行动-观察循环：代理在一个循环中工作：
# 3.1 思考(Thought)：LLM思考如何解决问题，并决定是否需要工具
# 3.2 行动(Action)： 如果需要工具，LLM会生成一个工具调用(工具名称和参数)
# 3.3 观察(Observation)：应用程序执行工具，并将工具的输出(观察结果)返回给LLM
# 3.4 重复：LLM再次思考，结合新的观察结果，决定下一步行动，知道问题解决或达到停止条件
#
#
# 代理的组件
# 1.LLM/ChatModel：代理的推理核心，通常是ChatOpenAI或其他支持函数调用的模型
# 2.Tools：代理可以使用的函数集合
# 3.AgentExecutor：驱动代理的运行时，它负责管理行动-观察循环、执行工具，并处理错误
#
# LangChain代理类型
# 1.create_react_agent:基于ReAct(Reasoning + Acting)框架。LLM会生成一个思考(Thought)来解释其推理过程，然后生成一个行动(Action)来调用工具
# 最后观察(Observation)结果
# 2.create_openai_tools_agetn:利用OpenAI模型强大的函数调用能力。模型直接生成工具调用，而不需要明确的思考步骤
# 3.create_json_agent:适用于需要模型输出JSON格式工具调用的场景
#
# 为什么使用代理
# 1.解决复杂问题：代理能够处理多步骤、需要外部信息或操作的复杂任务
# 2.动态决策：代理可以根据实时信息动态调整期计划，而不是遵循预设的固定路径
# 3.增强LLM能力：代理将LLM的语言理解和生成能力与外部工具的强大功能结合起来，极大的扩展了LLM的应用范围
#
# TODO:补充示例

from langchain.agents import AgentType, Tool, initialize_agent
from langchain_openai import ChatOpenAI
from llm_client_asher.llm_config import Config
from pydantic import SecretStr

cfg = Config()


def calculator_tool(query: str) -> str:
    """简单的计算工具"""
    try:
        return str(eval(query))
    except Exception as e:
        return f"Error: {str(e)}"


def example1():
    # 包装工具
    calculator = Tool(
        name="Calculator",
        func=calculator_tool,
        description="执行简单数据表达式计算",
    )
    if cfg.OPENAI_KEY_V4 is None:
        raise ValueError("OPENAI_KEY_V4 is not set")
    llm = ChatOpenAI(
        base_url=cfg.OPENAI_BASE_URL,
        api_key=SecretStr(cfg.OPENAI_KEY_V4),
        model="gpt-4-turbo",
    )
    # 初始化一个带有思考和行动的ReACT代理
    agent = initialize_agent(
        tools=[calculator],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    question = "请帮我计算一下 12 * (3 + 5)"

    # 运行agent
    response = agent.invoke({"input": question})
    print(response)


if __name__ == "__main__":
    example1()
