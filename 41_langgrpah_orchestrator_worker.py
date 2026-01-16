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
import os

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, SecretStr
from sqlalchemy.sql import desc
from typing_extensions import List, Literal, TypedDict

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


#
