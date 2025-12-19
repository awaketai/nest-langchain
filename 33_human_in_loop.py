# Human-in-the-loop(HITL):此中间件让你能够在爱智能体调用工具时加入人工监督。当模型提出一个可能需要审查的操作
# # 例如：写入文件或者SQL，中间件会根据可配置的策略来检查每一次工具调用，
# 如果需要人工介入，中间件会发出中断信号，暂停执行。借助LangGraph的持计划层，图状态会被安全保存，因此
# 执行可以暂停并在稍后恢复，人工决定下一步：approve/edit(修改后再运行)/reject
#
# 决策类型：
# approve: 同意执行，
# edit: 修改后再执行
# reject: 拒绝执行

import os
from logging import lastResort
from mimetypes import inited
from typing import Annotated, Literal

from aiohttp.web_routedef import delete
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.tools.render import ToolsRenderer
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.chat_agent_executor import create_tool_calling_executor
from numpy.random.mtrand import f
from pydantic import SecretStr
from typing_extensions import TypedDict

load_dotenv()

OPEN_API_URL = os.environ.get("OPEN_API_URL")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")


@tool
def write_file(filename: str, content: str) -> str:
    """写入文件到磁盘，这是一个敏感操作，需要人工审核"""
    # 文件写入逻辑
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return "文件写入成功"


@tool
def delete_file(filename: str) -> str:
    """删除文件，这是一个危险操作，需要人工审核"""
    if os.path.exists(filename):
        os.remove(filename)
        return "文件删除成功"
    else:
        return "文件不存在"


@tool
def execute_sql(query: str) -> str:
    """执行SQL查询，这是一个人工操作，需要人工审核"""
    return f"这是搜索 {query} 的结果"


@tool
def search_web(query: str) -> str:
    "搜索网页，这个一个安全操作，不需要人工审核"
    return f"这是搜索网页 {query} 的结果"


# 定义图的状态
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 定义需要人工审核的工具列表
TOOLS_REQUIRING_APPROVAL = ["write_file", "delete_file", "execute_sql"]


# 创建智能体节点
def create_agent_node(tools):
    """创建智能体节点"""
    if OPEN_API_KEY is None:
        raise ValueError("OPENAI_KEY_V4 is not set")
    model = ChatOpenAI(
        base_url=OPEN_API_URL,
        api_key=SecretStr(OPEN_API_KEY),
    )

    model_with_tools = model.bind_tools(tools)

    def call_model(state: State):
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    return call_model


# 创建人工审核节点
def human_review_node(state: State):
    """人工审核节点 - 检查是否有需要审核的工具调用
    这个节点会中断执行，等待人工决策
    """
    messages = state["messages"]
    last_message = messages[-1]

    # 检查最后一条消息是否包含工具调用
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name", "")
            if tool_name in TOOLS_REQUIRING_APPROVAL:
                # 需要人工审核，触发中断
                print(f" 检测到敏感操作：{tool_name} \n")
                print(f" 工具参数: {tool_call.get('args', {})} \n")
                print(" 请审核次操作 (approve/edit/reject) \n")

                # 在LangGraph中，这会触发一个中断点
                # 实际应用中会使用 interrupt() 函数
                break

    return state


# 路由函数，决定下一步走向
def should_continue(state: State) -> Literal["tools", "human_review", "end"]:
    """决定是继续执行工具还是需要人工审核"""
    messages = state["messages"]
    last_message = messages[-1]

    # 如果有工具调用
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # 检测是否需要人工审核
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name", "")
            if tool_name in TOOLS_REQUIRING_APPROVAL:
                return "human_review"

        # 不需要人工审核的工具直接执行
        return "tools"

    # 没有工具调用，结束
    return "end"


def create_hitl_graph():
    """创建带有 HUman-in-the-Loop 的图"""
    # 定义工具
    tools = [write_file, delete_file, execute_sql, search_web]
    # 创建图
    workflow = StateGraph(State)
    # 添加节点
    workflow.add_node("agent", create_agent_node(tools))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("human_review", human_review_node)

    # 添加边
    workflow.add_edge(START, "agent")

    # 添加条件边
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "human_review": "human_review", "end": END},
    )
    # 工具执行后回到 agent
    workflow.add_edge("tools", "agent")
    # 人工审核后也回到 tools (如果批准)，或其他处理
    workflow.add_edge("human_review", "tools")
    # 使用内存检查保存状态
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_with_approval():
    """运行带有人工审核的示例"""
    graph = create_hitl_graph()
    # 配置
    config = {"configurable": {"thread_id": "hitl-example-1"}}
    # 初始输入
    initial_input = {
        "messages": [
            ("user", "请帮我创建一个名为 test.txt 的文件，内容是 'Hello,World!'")
        ]
    }
    print("-" * 30)
    print("\n 示例1：带人工审核的文件写入操作\n")
    # 流式执行
    for event in graph.stream(initial_input, config, stream_mode="values"):
        if "messages" in event:
            last_message = event["messages"][-1]
            if hasattr(last_message, "content") and last_message.content:
                print(f" {last_message.content}")
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    print(f" 工具调用: {tool_call.get('name')}")
                    print(f" 参数: {tool_call.get('args')}")


def run_without_approval():
    """
    运行不需要审核的示例
    """
    graph = create_hitl_graph()
    # 配置
    config = {"configurable": {"thread_id": "hitl-example-2"}}
    # 初始输入
    initial_input = {"messages": [("user", "请帮我搜索一下Python最新版本的信息")]}
    print("-" * 30)
    print("\n 示例2：不需要审核的搜索操作\n")

    for event in graph.stream(initial_input, config, stream_mode="values"):
        if "messages" in event:
            last_message = event["messages"][-1]
            if hasattr(last_message, "content") and last_message.content:
                print(f" {last_message.content}")


def simulate_approval_workflow():
    """模拟完整的审核工作流"""
    print("-" * 30)
    print("示例3：模拟人工审核工作流\n")

    # 模拟场景
    scenarios = [
        {
            "action": "approve",
            "tool": "write_file",
            "args": {"filename": "approved.txt", "content": "This is Approved content"},
            "description": "批准执行",
        },
        {
            "action": "edit",
            "tool": "write_file",
            "args": {"filename": "oritinal.txt", "content": "This is Oritinal content"},
            "edit_args": {
                "filename": "modified.txt",
                "content": "This is Modified content",
            },
            "description": "修改参数后执行",
        },
        {
            "action": "reject",
            "tool": "delete_file",
            "args": {"filename": "important.txt"},
            "description": "拒绝执行",
        },
    ]
    for i, scenario in enumerate(scenarios, 1):
        print(f" 场景 {i}: {scenario['description']} \n")
        print(f" 工具: {scenario['tool']} \n")
        print(f" 参数: {scenario['args']} \n")
        if scenario["action"] == "approve":
            print(" 人工决策：批准 \n")
            print("执行工具 \n")

        elif scenario["action"] == "edit":
            print(" 人工决策：修改参数 \n")
            print("执行工具 \n")

        elif scenario["action"] == "reject":
            print(" 人工决策：拒绝 \n")
            print("执行工具 \n")


def demonstrate_interrupt_resume():
    """演示中断和恢复功能"""

    print("\n" + "=" * 60)
    print("示例 4: 中断和恢复执行")
    print("=" * 60)

    print("""
    在实际应用中，Human-in-the-Loop 的工作流程如下：

    1. 智能体开始执行任务
    2. 检测到需要审核的工具调用
    3. 系统触发中断（interrupt），保存当前状态
    4. 等待人工审核决策
    5. 根据决策：
       - approve: 从中断点恢复执行
       - edit: 修改参数后恢复执行
       - reject: 终止执行或返回错误信息

    关键代码模式：

    # 在工具执行前中断
    from langgraph.types import interrupt

    def human_approval_node(state):
        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls

        # 请求人工审核
        decision = interrupt({
            "tool_calls": tool_calls,
            "question": "是否批准这些工具调用？"
        })

        # 根据决策处理
        if decision == "approve":
            return state
        elif decision == "reject":
            raise ValueError("操作被拒绝")
        elif "edit" in decision:
            # 修改工具调用参数
            modified_state = modify_tool_calls(state, decision["new_args"])
            return modified_state

    # 恢复执行
    graph.invoke(None, config, resume_value="approve")
    """)


if __name__ == "__main__":
    # run_with_approval()
    # run_without_approval()
    # simulate_approval_workflow()
    # demonstrate_interrupt_resume()

    print("\n" + "=" * 60)
    print("💡 使用提示:")
    print("=" * 60)
    print("""
    1. 配置敏感工具列表 (TOOLS_REQUIRING_APPROVAL)
    2. 使用 MemorySaver 或其他检查点保存器持久化状态
    3. 在需要审核的节点使用 interrupt() 触发中断
    4. 使用 graph.invoke(..., resume_value=...) 恢复执行
    5. 可以实现 Web UI 让用户进行审核决策

    安全建议:
    - 对所有写操作 (文件、数据库) 启用审核
    - 对删除操作启用审核
    - 对外部 API 调用考虑启用审核
    - 记录所有审核决策用于审计
    - 实现超时机制，避免无限期等待
    """)
