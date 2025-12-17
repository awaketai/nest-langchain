# LangGraph是LangChain生态系统中的一个库，专门用于构建有状态、多参与者的LLM应用程序
# 它扩展了LangChain的概念，允许以图形(Graph)的形式定义复杂的LLM工作流，其中包含循环、条件逻辑和
# 多个代理之间的工作，这使得构建更健壮、更智能的代理系统成为可能
#
# 核心概念
# 1.图(Graph)：LangGraph的核心抽象，由节点(Nodes)和边(Edges)组成，节点代表计算步骤(如调用LLM、使用工具)，边定义了这些步骤之间的流转
# 2.节点(Nodes)：图中的一个处理单元，可以是LLM调用、工具调用、自定义函数等，节点可以有输入和输出
# 3.边(Edges)：连接节点，定义了数据和控制流。边可以是条件性的，允许根据节点输出进行动态路由
# 4.状态(State)：LangGraph的一个关键特性是其有状态性。图在执行过程中维护一个共享状态，节点可以读取和更新这个状态
# 4.代理(Agents)：LangGraph非常适合构建复杂的代理系统，其中多个代理可以协作解决问题，每个代理都是图中的一个或多个节点
#
# 用途：LangGraph适用于构建一下类型的应用程序：
# 1.复杂代理工作流：需要多步推理、工具使用、循环和条件逻辑的代理
# 2.多代理协作：多个LLM代理之间需要相互通信和协作来完成任务
# 3.有状态对话系统：需要维护复杂对话状态并根据状态动态调整行为的聊天机器人
# 4.自主代理：能够自我纠正、规划和执行复杂任务的代理
#
# 工作原理
# 在LangGraph中构建应用程序通常涉及一下步骤
# 1.自定义图状态：确定图在执行过程中需要维护哪些信息
# 2.定义节点：为图中的每个计算步骤定义一个节点。这可以是调用LLM、执行工具或运行自定义Python函数
# 3.定义边：连接节点，指定数据如何从一个节点流向另一个节点，可以定义条件边来实现动态路由
# 4.编译图：将定义的节点和边编译成一个可执行的图
# 5.运行图：通过图的入口点运行图，并观察状态的变化和输出
#
# 示例：
# 使用LangGraph构建一个最简单的有状态图，其中包含一个LLM调用节点和一个自定义处理节点
# 核心概念：
# GraphState：定义了图的共享状态，本例包含messages(对话历史)和user_input
# - messages:使用Annotated[List[BaseMessage]],lambda x,y: x + y来定义，标识消息列表是可累加的，每次更新都会将新消息添加到现有列表中
# call_llm节点：负责调用ChatOpenAI模型，根据当前状态中的messages生成LLM响应，并将响应作为新消息添加到状态中
# simple_processor节点：一个自定义处理函数，它从装填中提取最新的用户输入，并将其转换为大写形式，然后将处理结果作为新的HUmanMessaga添加到状态中
# StateGraph:LangGraph的核心类，用于定义图的结构
# add_node:向图中添加节点
# set_entry_point:设置图的起始节点
# add_edge:定义节点之间的流转
# complie():将定义的图编译成一个可执行的应用程序
# invoke():运行编译后的图，传入初始状态
#
#

import os
from mimetypes import inited
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import SecretStr
from sqlalchemy.sql.operators import mul

load_dotenv()

OPEN_API_URL = os.environ.get("OPEN_API_URL")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")


if OPEN_API_KEY is None:
    raise ValueError("OPENAI_KEY_V4 is not set")
llm = ChatOpenAI(
    base_url=OPEN_API_URL,
    api_key=SecretStr(OPEN_API_KEY),
    model="gpt-3.5-turbo",
    temperature=0,
)


# 定义状态类型
class GraphState(TypedDict):
    message: str
    llm_response: str
    processed_result: str


def llm_node(state: GraphState) -> GraphState:
    """
    llm调用节点
    """

    response = llm.invoke(state["message"])
    return {
        **state,
        "llm_response": response.content,
    }


# 自定义处理节点
def process_node(state: GraphState) -> GraphState:
    """
    对LLM响应进行自定义处理
    """
    llm_response = state["llm_response"]
    # 简单处理逻辑：添加前缀和后缀，统计字数
    word_count = len(llm_response.split())
    processed = f"[处理结果] {llm_response}，字数：{word_count}"

    return {
        **state,
        "processed_result": processed,
    }


# 构建图
def create_graph():
    # 创建状态图
    workflow = StateGraph(GraphState)
    # 添加节点
    workflow.add_node("llm", llm_node)
    workflow.add_node("process", process_node)

    # 设置入口点
    workflow.set_entry_point("llm")
    # 添加边：定义节点之间的连接
    workflow.add_edge("llm", "process")
    workflow.add_edge("process", END)

    # 编译图
    app = workflow.compile()
    return app


def main():
    app = create_graph()
    # 初始状态
    initial_state = {
        "message": "用一句话解释什么是人工智能",
        "llm_response": "",
        "processed_result": "",
    }
    # 执行图
    result = app.invoke(initial_state)
    print("原始消息:", result["message"])
    print("\nLLM 响应:", result["llm_response"])
    print("\n处理结果:", result["processed_result"])


# 定义状态类型，包含输入，两个代理的输出
class MultiAgentState(TypedDict):
    user_input: str
    agent1_summary: str
    agent2_advice: str


def agent1_node(state: MultiAgentState) -> MultiAgentState:
    """
    第一个代理：负责总结用户输入内容
    """
    prompt = f"请用一句简短的话总结以下内容: {state['user_input']}"
    response = llm.invoke(prompt)
    return {
        **state,
        "agent1_summary": response.content,
    }


def agent2_node(state: MultiAgentState) -> MultiAgentState:
    """
    第二个代理，根据第一个代理的总结生成建议
    """
    prompt = f"基于这段总结，给出改进建议: {state['agent1_summary']}"
    response = llm.invoke(prompt)
    return {
        **state,
        "agent2_advice": response.content,
    }


def multi_agent():
    graph = StateGraph(MultiAgentState)

    # 添加两个代理节点
    graph.add_node("agent1", agent1_node)
    graph.add_node("agent2", agent2_node)

    # 入口是agent1开始处理
    graph.set_entry_point("agent1")

    # 定义数据流，agent1的输出作为agetn2的输入
    graph.add_edge("agent1", "agent2")
    graph.add_edge("agent2", END)

    # 编译图
    app = graph.compile()
    return app


def multi_agent_main():
    app = multi_agent()
    initial_state = {
        "user_input": "LangGraph 是一个非常好用的库，可以帮助开发者构建复杂的有状态多代理工作流。",
        "agent1_summary": "",
        "agent2_advice": "",
    }
    result = app.invoke(initial_state)
    print("用户输入:", initial_state["user_input"])
    print("代理1总结:", result["agent1_summary"])
    print("代理2建议:", result["agent2_advice"])


if __name__ == "__main__":
    # main()
    multi_agent_main()
