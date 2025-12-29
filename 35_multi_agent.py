# Multi-agent 系统用来协调各种专门的组件来处理复杂的流程。
# https://docs.langchain.com/oss/python/langchain/multi-agent/skills
# 但是并不是每项复杂的任务都需要这种模式，只要具备恰当(有时候是动态)的工具，一个单一的agent也可实现相似的效果
#
#
# Skills：在 skill 架构中，特定的能力被封装为 "skill" ，以增强 agent 的行为。skill是主要的 prompt-driven 专业模块
# agent 可按需调用
#
#
import operator
import os
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr
from typing_extensions import TypedDict


class Skill(TypedDict):
    name: str
    description: str
    content: str


SKILLS: list[Skill] = [
    {
        "name": "sales_analytics",
        "description": "Database schema and business logic for sales data analysis including customers, orders, and revenue.",
        "content": """# Sales Analytics Schema

## Tables

### customers
- customer_id (PRIMARY KEY)
- name
- email
- signup_date
- status (active/inactive)
- customer_tier (bronze/silver/gold/platinum)

### orders
- order_id (PRIMARY KEY)
- customer_id (FOREIGN KEY -> customers)
- order_date
- status (pending/completed/cancelled/refunded)
- total_amount
- sales_region (north/south/east/west)

### order_items
- item_id (PRIMARY KEY)
- order_id (FOREIGN KEY -> orders)
- product_id
- quantity
- unit_price
- discount_percent

## Business Logic

**Active customers**: status = 'active' AND signup_date <= CURRENT_DATE - INTERVAL '90 days'

**Revenue calculation**: Only count orders with status = 'completed'. Use total_amount from orders table, which already accounts for discounts.

**Customer lifetime value (CLV)**: Sum of all completed order amounts for a customer.

**High-value orders**: Orders with total_amount > 1000

## Example Query

-- Get top 10 customers by revenue in the last quarter
SELECT
    c.customer_id,
    c.name,
    c.customer_tier,
    SUM(o.total_amount) as total_revenue
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
  AND o.order_date >= CURRENT_DATE - INTERVAL '3 months'
GROUP BY c.customer_id, c.name, c.customer_tier
ORDER BY total_revenue DESC
LIMIT 10;
""",
    },
    {
        "name": "inventory_management",
        "description": "Database schema and business logic for inventory tracking including products, warehouses, and stock levels.",
        "content": """# Inventory Management Schema

## Tables

### products
- product_id (PRIMARY KEY)
- product_name
- sku
- category
- unit_cost
- reorder_point (minimum stock level before reordering)
- discontinued (boolean)

### warehouses
- warehouse_id (PRIMARY KEY)
- warehouse_name
- location
- capacity

### inventory
- inventory_id (PRIMARY KEY)
- product_id (FOREIGN KEY -> products)
- warehouse_id (FOREIGN KEY -> warehouses)
- quantity_on_hand
- last_updated

### stock_movements
- movement_id (PRIMARY KEY)
- product_id (FOREIGN KEY -> products)
- warehouse_id (FOREIGN KEY -> warehouses)
- movement_type (inbound/outbound/transfer/adjustment)
- quantity (positive for inbound, negative for outbound)
- movement_date
- reference_number

## Business Logic

**Available stock**: quantity_on_hand from inventory table where quantity_on_hand > 0

**Products needing reorder**: Products where total quantity_on_hand across all warehouses is less than or equal to the product's reorder_point

**Active products only**: Exclude products where discontinued = true unless specifically analyzing discontinued items

**Stock valuation**: quantity_on_hand * unit_cost for each product

## Example Query

-- Find products below reorder point across all warehouses
SELECT
    p.product_id,
    p.product_name,
    p.reorder_point,
    SUM(i.quantity_on_hand) as total_stock,
    p.unit_cost,
    (p.reorder_point - SUM(i.quantity_on_hand)) as units_to_reorder
FROM products p
JOIN inventory i ON p.product_id = i.product_id
WHERE p.discontinued = false
GROUP BY p.product_id, p.product_name, p.reorder_point, p.unit_cost
HAVING SUM(i.quantity_on_hand) <= p.reorder_point
ORDER BY units_to_reorder DESC;
""",
    },
]


# 创建skills loading tool
@tool
def load_skills(skill_name: str) -> str:
    """Load the full content of a skill into the agent's context.

    Use this when you need detailed information about how to handle a specific
    type of request. This will provide you with comprehensive instructions,
    policies, and guidelines for the skill area.

    Args:
        skill_name: The name of the skill to load (e.g., "expense_reporting", "travel_booking")
    """
    # Find and return the requested skill
    for skill in SKILLS:
        if skill["name"] == skill_name:
            return f"Loaded skill: {skill_name}\n\n{skill['content']}"

    # Skill not found
    available = ", ".join(s["name"] for s in SKILLS)
    return f"Skill '{skill_name}' not found. Available skills: {available}"


# build skill middleware
# 由于当前版本(0.3.x)没有 middleware 和 create_agent API
# 我们使用 LangGraph + 自定义 prompt 的方式实现相同功能


# 定义 Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def create_skill_aware_agent(model, tools, system_prompt: str):
    """创建一个支持 skill 的 agent

    Args:
        model: LLM 模型
        tools: 工具列表（包括 load_skills 工具）
        system_prompt: 基础系统提示

    Returns:
        编译后的 LangGraph agent
    """

    # 构建 skills 列表文本
    skills_list = []
    for skill in SKILLS:
        skills_list.append(f"- **{skill['name']}**: {skill['description']}")
    skills_prompt = "\n".join(skills_list)

    # 组合完整的系统提示
    full_system_prompt = (
        f"{system_prompt}\n\n"
        f"## Available Skills\n\n{skills_prompt}\n\n"
        "Use the load_skills tool when you need detailed information "
        "about handling a specific type of request."
    )

    # 绑定工具到模型
    model_with_tools = model.bind_tools(tools)

    # 定义 agent 节点
    def call_model(state: AgentState):
        messages = state["messages"]

        # 在第一条消息前添加系统提示
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=full_system_prompt)] + list(messages)

        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    # 定义路由函数
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]

        # 如果没有工具调用，结束
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"
        return "continue"

    # 构建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    # 设置入口点
    workflow.set_entry_point("agent")

    # 添加条件边
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    # 工具执行后回到 agent
    workflow.add_edge("tools", "agent")

    # 编译图
    return workflow.compile(checkpointer=MemorySaver())


load_dotenv()

OPEN_API_URL = os.environ.get("OPEN_API_URL")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")
# 使用示例
if __name__ == "__main__":
    # 选择你的模型（根据你的环境调整）
    # from langchain_openai import ChatOpenAI
    # model = ChatOpenAI(model="gpt-4", temperature=0)

    if OPEN_API_KEY is None:
        raise ValueError("OPENAI_KEY_V4 is not set")
    model = ChatOpenAI(
        base_url=OPEN_API_URL,
        api_key=SecretStr(OPEN_API_KEY),
    )

    # 创建 agent
    agent = create_skill_aware_agent(
        model=model,
        tools=[load_skills],
        system_prompt=(
            "You are a SQL query assistant that helps users "
            "write queries against business databases."
        ),
    )

    # 测试对话
    config = {"configurable": {"thread_id": "test-thread-1"}}

    print("=== 测试 1: 询问销售数据查询 ===")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content="How do I find the top customers by revenue?")
            ]
        },
        config=config,
    )
    print(result["messages"][-1].content)

    print("\n=== 测试 2: 询问库存相关查询 ===")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content="Show me products that need to be reordered")
            ]
        },
        config=config,
    )
    print(result["messages"][-1].content)

    print("\n=== 测试 3: 直接使用 load_skills 工具 ===")
    result = agent.invoke(
        {"messages": [HumanMessage(content="Load the sales_analytics skill for me")]},
        config=config,
    )
    print(result["messages"][-1].content)
