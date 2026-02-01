import operator
import os
from typing import Literal

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import SecretStr
from typing_extensions import Annotated, TypedDict

load_dotenv()

OPEN_API_URL = os.environ.get("OPEN_API_URL")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")

if OPEN_API_KEY is None:
    raise ValueError("OPENAI_KEY_V4 is not set")

model = ChatOpenAI(
    base_url=OPEN_API_URL,
    api_key=SecretStr(OPEN_API_KEY),
    temperature=0,
)


# 1.Define tools
@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply `a` and `b`.
    Args:
        a (int): The first number.
        b (int): The second number.
    Returns:
        int: The product of `a` and `b`.
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """
    Add `a` and `b`.
    Args:
        a (int): The first number.
        b (int): The second number.
    Returns:
        int: The sum of `a` and `b`.
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """
    Divide `a` by `b`.
    Args:
        a (int): The numerator.
        b (int): The denominator.
    Returns:
        float: The quotient of `a` divided by `b`.
    """
    return a / b


tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


# 2.Define state
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# 3.定义模型节点


def llm_call(state: dict):
    """
    LLM decides whether to call a tool or not.
    """
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# 4.定义tool节点
def tool_node(state: dict):
    """
    Performs the tool call.
    """
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return {"messages": result}


# 5.定义结束逻辑
def should_continue(state: MessagesState) -> Literal["tool_name", END]:
    """
    Decide if we should continue the loop or sop based upon whether the LLM made a tool call.
    """
    messages = state["messages"]
    last_message = messages[-1]
    # 如果LLM 要进行工具调用
    if last_message.tool_calls:
        return "tool_node"

    return END


# 6.编译agent
# 构建workflow
agent_builder = StateGraph(MessagesState)
# 添加节点
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
# 添加边界，连接节点
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", END)
# compile
agent = agent_builder.compile()
# show agent

# from IPython.display import Image, display
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# 调用
messages = [HumanMessage(content="add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
