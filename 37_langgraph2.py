import operator
import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
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
model_with_tools = model.bind(tools=tools)


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
        "llm_calls": state["llm_calls", 0] + 1,
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
