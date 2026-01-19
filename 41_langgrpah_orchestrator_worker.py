# Orchestrator Worker 编排器
# 在编排器-工作及节点中，编排器的职责如下
# 1.将任务拆解为若干子任务
# 2.将子任务派给各个工作节点
# 3.整合工作节点的输出结果，生成最终成果
#
# Orchestrator Worker 工作流模式具备更高的灵活性，通常用于子任务无法像并行计算那样预先定义的场景
# 这种情况在代码编写或多文件内容更新类的工作流中尤为常见。例如，需要在数量不确定的文档中，更新多个 Python 库
# 的安装说明，就可以采用该架构模式
#
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


class Section(BaseModel):
    name: str = Field(description="Name for this section of the report")
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section",
    )


class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections of the report")


# 为大模型配置结构化输出的模式
plannder = model.with_structured_output(Sections)

# Send API 允许动态创建工作器节点，并向其发送特定的输入数据，每个工作器都拥有独立的状态，且所有工作期的输出都会写入一个共享状态键，供 orchestrator graph 使用。
# 这一机制能够让 orchestrator 能够获取所有工作器的输出，并将这些输出整合为最终结果，下方示例遍历每一个章节列表，同时借助 Send API 将每个部分分别发送至对应的工作器


# Graph state
class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    #
    final_report: str


# Worker state
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


# 节点
def orchestrator(state: State):
    """编排器 生成一个关于 report 的计划"""

    # generate queries
    report_sections = plannder.invoke(
        [
            SystemMessage(content="Generate a plan for the report"),
            HumanMessage(content="Here is the report topic: {}".format(state["topic"])),
        ]
    )

    return {"sections": report_sections.sections}


def llm_call(state: WorkerState):
    """工作节点书写关于 report 的内容"""
    # generate section
    section = model.invoke(
        [
            SystemMessage(
                content="Write a report section following the provided name and description.Include no preamble for each section. Use markdown formatting."
            ),
            HumanMessage(
                content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
            ),
        ]
    )
    # 将结果写入 completed section
    return {"completed_sections": [section.content]}


def synthesizer(state: State):
    """合成所有 report"""
    completed_sections = state["completed_sections"]
    # 将转换为字符串，作为最终结果
    completed_report_sections = "\n\n---\n\n".join(completed_sections)
    return {"final_report": completed_report_sections}


def assign_workers(state: State):
    """给每个计划的 section 分配工作节点"""
