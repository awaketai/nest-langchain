# 在LangChain中，工具(Tool)是代理(Agent)于外部世界交互的手段。
# 他们是LLM可以调用的函数，用于执行特定任务，例如搜索互联网、执行代码、查询数据库等
# 工具是赋予LLM行动能力的关键组件
#
# 工具：本质上是一个函数，有名称、描述和可选的参数定义
# 名称(Name):唯一的标识符，代理用它来引用工具
# 描述(Description):最重要的部分，这是代理用来决定何时以及如何使用工具的自然语言描述。
# 描述越清晰、越准确，代理就越能正确的使用工具
#
# 参数(Arguments):工具函数所需要的输入,LangChain鼓励使用Pydantic模型来定义参数的结构和类型
# 这有助于代理生成正确的参数
#
# 如何定义工具
# 1.使用@tool装饰器(推荐)，这是定义简单Python函数作为工具的最简单、推荐的方式
#
# 1.函数的名称(search)会自动成为工具的name
# 2.函数的docstring会自动成为工具的description
# 3.函数的类型会自动用于生成工具的args_schema
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search the web for the given query."""
    return f"Search results for {query}"


#
# 2.使用StructuredTool(用于更复杂的参数)
# 当工具需要更复杂的、结构化的输入时，可以使用StructuredTool并结合Pydantic模型来定义args_schema
#
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    query: str = Field(..., description="The query to search for")


@tool
def search_structured(input: SearchInput) -> str:
    """Search the web for the given query."""
    return f"Search results for {input.query}"


calculator_tool = StructuredTool.from_function(
    func=search_structured,
    name="search_structured",
    description="Search the web for the given query.",
    args_schema=SearchInput,
)

# 3.集成BaseTool（最灵活）
# 对于需要完全自定义行为(例如：异步执行、自定义错误处理)的工具，可以直接集成langchain_core.tools.BaseTool
#
# 代理的思考依赖于工具的描述，LLM会阅读这些描述，并根据他们来决定：
# 1.何时调用工具(用户的问题是否与工具的功能匹配)
# 2.如何调用工具(需要哪些参数？参数的含义是什么)
#
# LangChain常见的内置工具
# TavilySearchResults:互联网搜索
# PythonREPLTool:执行Python代码
# LLMMathChain:进行数学计算
# WikipediaQueryRun:查询维基百科
