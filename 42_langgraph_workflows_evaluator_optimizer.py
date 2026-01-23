# Evaluator optimizer
# 在评估-优化工作流中，一个大语言模型调用负责生成回复内容，另一个则负责对该回复进行评估，若评估器或人工介入判定回复需要优化，则会给出反馈信息，并重新生成回复内容。这一循环会持续进行，直到生成符合要求的回复
#
# 当某项任务具备明确的成功判定标准、但需要通过迭代来达成标准时，评估器-优化器工作流尤为普遍。例如，在两种文本语言的互译过程中，并非总能实现精确匹配；要生成语义完全对等的译文，往往需要经过数次迭代优化
#
import operator
import os

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field, SecretStr
from sqlalchemy.sql import desc
from typing_extensions import Annotated, List, Literal, TypedDict

load_dotenv()

OPEN_API_URL = os.environ.get("OPEN_API_URL")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")

if OPEN_API_KEY is None:
    raise ValueError("OPENAI_KEY_V4 is not set")

model = ChatOpenAI(
    base_url=OPEN_API_URL,
    api_key=SecretStr(OPEN_API_KEY),
    temperature=0,
    model="gpt-4o-mini",
)


class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str


class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not."
    )
    feedback: str = Field(
        description="If the joke is not funny,provide feedback on how to improve it."
    )


evaluator = model.with_structured_output(Feedback)


# 节点
def llm_call_generator(state: State):
    """LLM generate a joke"""
    if state.get("feedback"):
        msg = model.invoke(
            f"Write a joke about {state['topic']} but take into account the feedback:{state['feedback']}"
        )
    else:
        msg = model.invoke(f"Write a joke about {state['topic']}")

    return {"joke": msg}
