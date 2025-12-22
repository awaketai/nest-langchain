# MCP (Model Context Protocol)是一个开源的标准协议，定义了应用和上下文应该如何提供给LLM，LangChain代理可以使用langchain-mcp-adapters库来调用
# MCP服务定义的工具
#
#
import asyncio
import json
from typing import List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# 创建mcp服务实例
app = Server("custom-mcp-server")

# 模拟天气数据
WEATHER_DATA = {
    "北京": {"temperature": -5, "condition": "晴天", "humidity": 30},
    "上海": {"temperature": 8, "condition": "多云", "humidity": 60},
    "广州": {"temperature": 18, "condition": "阴天", "humidity": 75},
    "深圳": {"temperature": 20, "condition": "晴天", "humidity": 65},
}


# 模拟数据库
DATABASE = {
    "users": [
        {"id": 1, "name": "张三", "age": 28, "city": "北京"},
        {"id": 2, "name": "李四", "age": 32, "city": "上海"},
        {"id": 3, "name": "王五", "age": 25, "city": "广州"},
    ],
    "products": [
        {"id": 1, "name": "笔记本电脑", "price": 5999, "stock": 50},
        {"id": 2, "name": "手机", "price": 3999, "stock": 100},
        {"id": 3, "name": "平板电脑", "price": 2999, "stock": 30},
    ],
}


@app.list_tools()
async def list_tools() -> List[Tool]:
    """
    列出所有可用的工具
    """
    return [
        Tool(
            name="get_weather",
            description="获取指定城市的天气信息，支持的城市：北京、上海、广州、深圳",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海",
                    }
                },
                "required": ["city"],
            },
        ),
        Tool(
            name="calculator",
            description="执行基本的数学计算，支持加、减、乘、除运算",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "运算类型：add(加)、subtract(减)、multiply(乘)、divide(除)",
                    },
                    "a": {
                        "type": "number",
                        "description": "第一个数字",
                    },
                    "b": {
                        "type": "number",
                        "description": "第二个数字",
                    },
                },
                "required": ["operation", "a", "b"],
            },
        ),
        Tool(
            name="query_database",
            description="查询数据库中的数据。可以查询用户(users)或产品(products)表",
            inputSchema={
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "enum": ["users", "products"],
                        "description": "要查询的表名",
                    },
                    "filter_field": {
                        "type": "string",
                        "description": "筛选字段名（可选）",
                    },
                    "filter_value": {
                        "type": "string",
                        "description": "筛选值（可选）",
                    },
                },
                "required": ["table"],
            },
        ),
        Tool(
            name="text_analyzer",
            description="分析文本并返回统计信息，包括字符数、词数等",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要分析的文本内容",
                    }
                },
                "required": ["text"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """处理工具调用"""
    if name == "get_weather":
        city = arguments.get("city")
        if city in WEATHER_DATA:
            weather = WEATHER_DATA[city]
            result = (
                f"{city}的天气情况：\n"
                f"温度：{weather['temperature']}°C\n"
                f"天气：{weather['condition']}\n"
                f"湿度：{weather['humidity']}%"
            )
            return [TextContent(type="text", text=result)]
        else:
            return [
                TextContent(
                    type="text",
                    text=f"抱歉，暂不支持查询 {city} 的天气。支持的城市：北京、上海、广州、深圳",
                )
            ]

    elif name == "calculator":
        operation = arguments.get("operation")
        a = arguments.get("a")
        b = arguments.get("b")

        try:
            if operation == "add":
                result = a + b
                op_symbol = "+"
            elif operation == "subtract":
                result = a - b
                op_symbol = "-"
            elif operation == "multiply":
                result = a * b
                op_symbol = "×"
            elif operation == "divide":
                if b == 0:
                    return [TextContent(type="text", text="错误：除数不能为零")]
                result = a / b
                op_symbol = "÷"
            else:
                return [TextContent(type="text", text=f"不支持的运算：{operation}")]

            return [
                TextContent(
                    type="text", text=f"计算结果：{a} {op_symbol} {b} = {result}"
                )
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"计算错误：{str(e)}")]

    elif name == "query_database":
        table = arguments.get("table")
        filter_field = arguments.get("filter_field")
        filter_value = arguments.get("filter_value")

        if table not in DATABASE:
            return [TextContent(type="text", text=f"表 {table} 不存在")]

        data = DATABASE[table]

        # 如果有筛选条件
        if filter_field and filter_value:
            filtered_data = [
                item
                for item in data
                if str(item.get(filter_field, "")).lower() == str(filter_value).lower()
            ]
            result_text = (
                f"查询结果（{table} 表，筛选 {filter_field}={filter_value}）：\n"
            )
            result_text += json.dumps(filtered_data, ensure_ascii=False, indent=2)
        else:
            result_text = f"查询结果（{table} 表，全部数据）：\n"
            result_text += json.dumps(data, ensure_ascii=False, indent=2)

        return [TextContent(type="text", text=result_text)]

    elif name == "text_analyzer":
        text = arguments.get("text", "")

        # 分析文本
        char_count = len(text)
        word_count = len(text.split())
        line_count = len(text.split("\n"))

        # 统计中文字符
        chinese_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")

        result = (
            f"文本分析结果：\n"
            f"总字符数：{char_count}\n"
            f"单词数：{word_count}\n"
            f"行数：{line_count}\n"
            f"中文字符数：{chinese_count}"
        )

        return [TextContent(type="text", text=result)]

    else:
        return [TextContent(type="text", text=f"未知的工具：{name}")]


async def main():
    """启动MCP服务"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    print("启动自定义MCP 服务")
    asyncio.run(main())
