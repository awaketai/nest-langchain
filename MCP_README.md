# LangChain MCP Adapters 使用指南

本指南展示如何使用 `langchain-mcp-adapters` 包来连接和使用 Model Context Protocol (MCP) 服务器。

## 📋 目录

- [什么是 MCP](#什么是-mcp)
- [安装依赖](#安装依赖)
- [项目文件说明](#项目文件说明)
- [快速开始](#快速开始)
- [详细示例](#详细示例)
- [自定义 MCP 服务器](#自定义-mcp-服务器)
- [常见问题](#常见问题)

## 什么是 MCP

Model Context Protocol (MCP) 是一个开放协议，用于将 LLM 应用程序与数据源和工具连接起来。它提供了：

- **标准化接口**：统一的方式来暴露工具和资源
- **多种传输方式**：支持 stdio、SSE 等多种通信协议
- **可扩展性**：易于创建自定义服务器和工具

## 安装依赖

```bash
pip install langchain-mcp-adapters
pip install langchain-openai
pip install mcp
```

或使用项目的 requirements.txt：

```bash
pip install -r requirements.txt
```

## 项目文件说明

本项目包含两个主要文件：

### 1. `mcp_server_example.py` - 自定义 MCP 服务器

这是一个完整的 MCP 服务器实现，提供了以下工具：

- **get_weather**: 获取城市天气信息
- **calculator**: 执行基本数学运算
- **query_database**: 查询模拟数据库
- **text_analyzer**: 分析文本统计信息

### 2. `mcp_client_example.py` - LangChain 客户端

演示如何使用 LangChain 连接和调用 MCP 服务器的工具。

## 快速开始

### 步骤 1: 测试 MCP 服务器

首先，确保 MCP 服务器可以正常运行：

```bash
python mcp_server_example.py
```

服务器应该会启动并等待连接。

### 步骤 2: 配置 API 密钥

在 `mcp_client_example.py` 中配置您的 OpenAI API 密钥：

```python
import os

# 使用 OpenAI
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# 或使用 DeepSeek
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"
os.environ["OPENAI_API_KEY"] = "your-deepseek-api-key"
```

### 步骤 3: 运行客户端

```bash
python mcp_client_example.py
```

## 详细示例

### 示例 1: 简单的工具调用

```python
import asyncio
from langchain_mcp_adapters.client import create_mcp_client

async def simple_example():
    # 连接到 MCP 服务器
    async with create_mcp_client(
        command="python",
        args=["mcp_server_example.py"],
    ) as client:
        # 获取所有工具
        tools = await client.list_tools()
        print(f"加载了 {len(tools)} 个工具")
        
        # 直接调用工具
        for tool in tools:
            if tool.name == "get_weather":
                result = await tool.ainvoke({"city": "北京"})
                print(f"结果: {result}")
                break

asyncio.run(simple_example())
```

### 示例 2: 在 LangChain Agent 中使用

```python
import asyncio
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import create_mcp_client
from langchain_openai import ChatOpenAI

async def agent_example():
    async with create_mcp_client(
        command="python",
        args=["mcp_server_example.py"],
    ) as client:
        # 获取工具
        tools = await client.list_tools()
        
        # 创建 LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有用的助手，可以使用各种工具来帮助用户。"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # 创建 Agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
        )
        
        # 执行查询
        response = await agent_executor.ainvoke({
            "input": "北京的天气怎么样？"
        })
        print(response['output'])

asyncio.run(agent_example())
```

### 示例 3: 多工具协作

```python
async def multi_tool_example():
    async with create_mcp_client(
        command="python",
        args=["mcp_server_example.py"],
    ) as client:
        tools = await client.list_tools()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有用的助手。请使用工具回答问题。"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # 需要使用多个工具的复杂查询
        response = await agent_executor.ainvoke({
            "input": "计算 100 加 200，然后告诉我上海的天气"
        })
        print(response['output'])

asyncio.run(multi_tool_example())
```

## 自定义 MCP 服务器

### 创建新工具

在 `mcp_server_example.py` 中添加新工具：

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("my-custom-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="my_custom_tool",
            description="这是一个自定义工具",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "参数说明",
                    }
                },
                "required": ["param1"],
            },
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "my_custom_tool":
        param1 = arguments.get("param1")
        result = f"处理参数: {param1}"
        return [TextContent(type="text", text=result)]
```

### 工具设计最佳实践

1. **清晰的描述**：确保工具描述清楚地说明功能
2. **完善的参数定义**：使用 JSON Schema 定义参数类型和约束
3. **错误处理**：优雅地处理错误情况
4. **返回格式统一**：始终返回 `list[TextContent]`

## 连接不同类型的 MCP 服务器

### 1. Stdio 协议（本地进程）

```python
async with create_mcp_client(
    command="python",
    args=["mcp_server_example.py"],
) as client:
    tools = await client.list_tools()
```

### 2. Node.js 服务器

```python
async with create_mcp_client(
    command="node",
    args=["path/to/server.js"],
) as client:
    tools = await client.list_tools()
```

### 3. SSE 协议（HTTP）

```python
async with create_mcp_client(
    url="http://localhost:8000/sse",
) as client:
    tools = await client.list_tools()
```

## 常见问题

### Q1: 服务器无法连接

**解决方案**：
- 确保 MCP 服务器脚本路径正确
- 检查 Python 环境是否正确安装了 `mcp` 包
- 查看是否有端口冲突

### Q2: 工具调用失败

**解决方案**：
- 检查工具参数是否符合 schema 定义
- 查看服务器日志确认错误信息
- 确保 LLM 正确理解了工具描述

### Q3: Agent 不使用工具

**解决方案**：
- 使用支持函数调用的模型（如 gpt-4、gpt-3.5-turbo）
- 检查工具描述是否清晰
- 在提示中明确告诉 Agent 可以使用工具

### Q4: 如何调试 MCP 服务器

**方法**：
```python
# 在服务器中添加日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 在客户端中启用详细输出
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 启用详细日志
)
```

### Q5: 支持哪些 LLM

支持所有 LangChain 兼容的聊天模型，包括但不限于：
- OpenAI (GPT-4, GPT-3.5-turbo)
- DeepSeek
- Anthropic Claude
- Google Gemini
- 本地模型（通过 Ollama 等）

## 示例运行输出

```
╔═══════════════════════════════════════════════════════════╗
║  LangChain MCP Adapters 使用示例                          ║
╚═══════════════════════════════════════════════════════════╝

📡 连接到 MCP 服务器...
✅ 成功连接到 MCP 服务器

🔧 加载 MCP 工具...
✅ 成功加载 4 个工具:
   - get_weather: 获取指定城市的天气信息
   - calculator: 执行基本的数学计算
   - query_database: 查询数据库中的数据
   - text_analyzer: 分析文本并返回统计信息

【测试 1: 天气查询】
> Entering new AgentExecutor chain...
> Invoking tool: get_weather with {'city': '北京'}
北京的天气情况：
温度：-5°C
天气：晴天
湿度：30%

回答: 北京今天是晴天，温度为-5°C，湿度30%
```

## 资源链接

- [Model Context Protocol 官方文档](https://modelcontextprotocol.io/)
- [langchain-mcp-adapters GitHub](https://github.com/langchain-ai/langchain-mcp-adapters)
- [LangChain 官方文档](https://python.langchain.com/)

## 许可证

本示例代码采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

---

**注意**：在生产环境中使用时，请确保：
1. 妥善保管 API 密钥
2. 实现适当的错误处理
3. 添加必要的安全检查
4. 考虑并发和性能优化