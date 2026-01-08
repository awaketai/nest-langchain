# Workflows and agents
# 工作流具有预先设定的代码路径，并且被设计成按照特定顺序运行
# 代理动态决定流程和工具的使用方式

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph

load_dotenv()

OPEN_API_URL = os.environ.get("OPEN_API_URL")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")

if OPEN_API_KEY is None:
    raise ValueError("OPENAI_KEY_V4 is not set")

llm = ChatOpenAI(
    base_url=OPEN_API_URL,
    api_key=SecretStr(OPEN_API_KEY),
    temperature=0,
    model="gpt-3.5-turbo"
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
msg = llm_with_tools.invoke("What is 2 multiply 3?")
# 获取工具调用信息
response = msg.tool_calls
print(response)

# Graph State
class State(TypedDict):
    topic:str
    joke: str
    improved_joke: str
    final_joke:str

# Nodes
def general_joke(state: State):
    """
    首次调用
    """
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {'joke':msg.content}

def check_punchline(state: State):
    """
    Gate funciton to check if the joke has a punchline
    """
    if "?" in state['joke'] or "!" in state['joke']:
        return "Pass"
    return "Fail"

def improve_joke(state: State):
    """
    Second LLM call to improve the joke
    """
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}

def polish_joke(state: State):
    """
    Third LLM call for final polish
    """
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}

# 构建工作流
workflow = StateGraph(State)

# 添加节点
workflow.add_node("generate_joke",general_joke)
workflow.add_node("improve_joke",improve_joke)
workflow.add_node("polish_joke",polish_joke)

# 添加边界
workflow.add_edge(START,"generate_joke")
workflow.add_conditional_edges("generate_joke",check_punchline,{"Fail":"improve_joke","Pass":END})
workflow.add_edge("improve_joke","polish_joke")
workflow.add_edge("polish_joke",END)

# 编译
chain = workflow.compile()

state = chain.invoke({"topic":"dogs"})

print("initial joke:",state['joke'])
print("\n--- --- ---\n")
if "improved_joke" in state:
    print("Improved joke:")
    print(state["improved_joke"])
    print("\n--- --- ---\n")

    print("Final joke:")
    print(state["final_joke"])
else:
    print("Final joke:")
    print(state["joke"])
