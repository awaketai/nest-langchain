# 在许多实际应用中，我们需要为代理提供一组相互关联、服务于特定领域或任务的工具
# 例如，一个需要与SQL数据库交互的代理，可能需要“列出表”，“查看表结构”，“执行SQL查询”等多个工具
# 工具箱(Toolkit)正是为了解决这个问题而设计的
#
# 工具箱：LangChain提供的一种抽象，它将一组功能相关的工具打包在一起，通常保函费
# 1.多个BaseTool实例：这些工具共同提供了一个特定领域的功能
# 2.预设的提示和代理类型：许多工具还会提供或推荐与该工具箱配合使用的特定提示模板和代理类型，以确保最佳性能
#
# 为什么使用工具箱
# 1.简化集成：无需手动定义和管理每个工具，只需要实例化工具箱，它就会提供所有必要的工具
# 2.领域特定功能：工具箱专注于特定领域，如SQL数据库等，为代理提供了强大的领域知识和操作能力
# 3.最佳实践：工具箱通常由LangChain团队或社区团队维护，包含了于该领域交互的最佳实践
# 4.可扩展性：开发着可以创建自己的自定义工具箱，以封装特定于其应用程序的工具集
#
# LangChain提供的工具箱
# SQLDatabaseToolkit:用于与SQL数据库交互。它提供了执行SQL查询、列出表、查看表等结构工具
# PandasDataFrameToolkit: 允许代理与 Pandas DataFrame 交互，执行数据分析任务。
#
# 如何使用工具箱
# 1.实例化工具箱：根据需求实例化相应的工具箱
# 2.获取工具：工具箱会提供一个get_tools()方法，返回一个BaseTool实例的列表
# 3.创建代理：将这些工具传递给代理的构造函数(例如create_openai_tools_agent)
#
#
#

import os
import sqlite3
from marshal import load

from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from llm_client_asher.llm_config import Config
from pydantic import SecretStr

cfg = Config()

load_dotenv()

OPEN_API_URL = os.environ.get("OPEN_API_URL")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")


def prepare_simple_data():
    """使用LangChain的SQLDatabaseToolkit构建SQL数据库代理"""
    # 准备数据
    conn = sqlite3.connect("company.db")
    cursor = conn.cursor()
    # 创建员工表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary INTEGER NOT NULL,
            hire_date TEXT NOT NULL
        )
    """)

    # 创建部门表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            budget INTEGER NOT NULL,
            manager TEXT
        )
    """)

    # 插入示例数据
    employees_data = [
        (1, "张三", "技术部", 15000, "2020-01-15"),
        (2, "李四", "技术部", 18000, "2019-05-20"),
        (3, "王五", "销售部", 12000, "2021-03-10"),
        (4, "赵六", "销售部", 13000, "2020-11-05"),
        (5, "钱七", "人力资源部", 14000, "2018-07-30"),
        (6, "孙八", "技术部", 20000, "2017-02-14"),
    ]

    cursor.executemany(
        "INSERT OR REPLACE INTO employees VALUES (?, ?, ?, ?, ?)", employees_data
    )

    departments_data = [
        (1, "技术部", 500000, "孙八"),
        (2, "销售部", 300000, "王五"),
        (3, "人力资源部", 200000, "钱七"),
    ]

    cursor.executemany(
        "INSERT OR REPLACE INTO departments VALUES (?, ?, ?, ?)", departments_data
    )

    conn.commit()
    conn.close()
    print("✓ 示例数据库创建成功！")


def sql_database_toolkit():
    """使用LangChain的SQLDatabaseToolkit构建SQL数据库代理"""
    db = SQLDatabase.from_uri("sqlite:///company.db")
    print(f"连接到数据库，包含以下表:{db.get_usable_table_names}")

    # 初始化llm
    if OPEN_API_KEY is None:
        raise ValueError("OPENAI_KEY_V4 is not set")

    llm = ChatOpenAI(
        base_url=OPEN_API_URL,
        api_key=SecretStr(OPEN_API_KEY),
        model="gpt-4",
    )

    # 创建sql工具包
    toolkit = SQLDatabaseToolkit(
        db=db,
        llm=llm,
    )
    # 创建sql代理
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,  # 处理解析错误
        max_iterations=5,
        max_execution_time=60,
    )

    return agent_executor


def run_queries():
    prepare_simple_data()

    queries = [
        "数据库中有哪些表？",
        "技术部有多少员工？",
        "工资最高的员工是谁？他的工资是多少？",
        "每个部门的平均工资是多少？",
        "哪个部门的预算最高？",
        "列出所有在2020年之后入职的员工姓名和部门",
    ]
    print("-" * 30)
    agent_executor = sql_database_toolkit()

    for i, query in enumerate(queries, 1):
        print(f"问题:{query}\n")
        try:
            response = agent_executor.invoke({"input": query})
            print(f"答案:{response['output']}")
        except Exception as e:
            print(f"错误:{e}")


if __name__ == "__main__":
    run_queries()
