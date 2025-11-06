# 函数调用(工具调用)：允许开发者向模型提供一系列可供其调用的工具(即函数)，让模型能够与外部世界进行交互，获取实时信息或执行特定任务
# 函数调用：
# 传统的语言模型只能处理文本：输入文本，输出文档，他们无法知道今天的日期，无法查询数据库，也无法预订机票
# 函数调用打破了这一限制
# 过程：
# 1.提供工具：向模型描述一系列它可以使用的工具，每个工具都有一个名称、一段描述以及清晰的参数定义
# 2.模型决策：当向模型提问时，模型会分析问题，如果它认为使用某个工具能更好的回答问题，它不会直接生成文本答案
# 相反，它会生成一个特殊的JSON对象，其中包含它想要调用的函数名和它推断出的参数
#
# 3.执行工具：你的应用程序代码会捕获这个json对象，并实际执行对应的函数(例如：调用一个天气API)
# 4.返回结果：你将函数执行的结果包装成一个ToolMessage，然后再次发送给模型
# 5.最终答案：模型接收到工具的执行结果后，会用自然语言将其总结成一个最终的、人类可读的答案

# LangChain中的函数调用
# 1.定义工具时可以用Pydantic或@tool装饰器把参数讲明白
# 2.需要让模型认识这些工具时，只要通过.bind_tools()或.bind_functions()绑定
# 3.当模型回应时，AIMessage里的tool_calls会列出它想要的函数名和参数

import os
import json
from pprint import pprint
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from llm_client_asher.config import Config

cfg = Config()


class GetWeatherArgs(BaseModel):
    location: str = Field(description="需要查询天气的城市名称，例如：北京")


@tool(args_schema=GetWeatherArgs)
def get_weather(location: str) -> str:
    """
    用于获取指定城市当前的天气信息
    """
    if "北京" in location:
        return json.dumps(
            {"location": "北京", "temperature": "25°C", "condition": "晴"}
        )
    elif "上海" in location:
        return json.dumps(
            {"location": "上海", "temperature": "28°C", "condition": "多云"}
        )
    elif "Shanghai" in location.lower():
        return json.dumps(
            {"location": "上海", "temperature": "28°C", "condition": "多云"}
        )
    else:
        return json.dumps(
            {"location": location, "temperature": "未知", "condition": "未知"}
        )


def main():
    """
    本实例将演示一个完整的端到端的函数调用流程
    1.模型决定调用工具
    2.我们解析该调用，并执行对应的函数
    3.我们将函数返回的结果包装成`ToolMessage`
    4.我们将`ToolMessage`连同对话历史一起再次发送给模型
    5.模型根据工具的返回结果，生成最终的自然语言会带
    """
    model = ChatOpenAI(
        base_url=cfg.OPENAI_BASE_URL,
        api_key=cfg.OPENAI_KEY_V4,
    )
    model_with_tools = model.bind_tools([get_weather])

    messages = [HumanMessage(content="上海今天天气怎么样？")]
    print("----步骤1 & 2：模型决策---\n")
    first_response = model_with_tools.invoke(messages)
    messages.append(first_response)

    print(f"模型返回的AIMessage包含 tool_calls: {first_response.tool_calls}")
    print("-" * 30)
    print("--- 步骤 3 & 4：在应用侧执行工具---\n")
    if not first_response.tool_calls:
        print("模型没有调用工具，流程结束\n")
        return
    for tool_call in first_response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        print(f"检测到工具调用: {tool_name}({tool_args})\n")
        if tool_name == get_weather.name:
            tool_output = get_weather.invoke(tool_args)
            print(f"工具执行结果:{tool_output}\n")
            messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call_id))
        else:
            print(f"未知的工具:{tool_name}\n")
            messages.append(
                ToolMessage(
                    content=f"未知的工具:{tool_name}", tool_call_id=tool_call_id
                )
            )
    print("当前消息历史:\n")
    print(messages)
    print("-" * 30)
    print("---步骤5：模型生成最终回答:---")
    final_response = model_with_tools.invoke(messages)

    print("最终的自然语言回答\n")
    print(f"类型{type(final_response)}")
    print(f"内容{final_response.content}")


if __name__ == "__main__":
    main()
