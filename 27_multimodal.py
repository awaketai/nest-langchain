# LangChain提供了与多模态模型交互的能力，允许您处理和生成包含文本、图像、音频等多种模态的数据
# 这使得构建更丰富、更具上下文感知的应用程序成为可能
#
#
# 核心概念
# 多模态提示(Multimodal Prompts)：能够将不同类型的数据（例如、文本和图像）组合到一个提示中，发送给多模态模型进行处理
# 多模态模型（Multimodal Models）：能够理解和生成多种模态数据的模型
#
# 用途：
# 1.图像描述与问答：向模型提供图像和文本问题，让模型生成图像描述或回答与图像相关的问题
# 2.视觉内容理解：分析图像内容并提取关键信息
# 3.多模态聊天机器人：构建能够理解用户输入的文本和图像，并以多模态方式响应的聊天机器人
#
# 工作原理
# 在LangChain中，与多模态模型交互通常涉及一下步骤
# 1.准备多模态输入：将不同模态的数据（例如，文本字符串和图像的Base64编码）组织成模型可以理解的格式
# 2.构建多模态提示：使用LangChain的提示模板功能，将多模态输入嵌入到提示中
# 3.调用多模态模型：将构建好的多模态提示发送给支持多模态的LLM或聊天模型
# 4.处理多模态输出：接收并解析模型的响应，其中可能包含文本、图像或其他模态的数据
#
# 示例
# 1.多模态提示：演示了如何在LangChain中构建和使用多模态提示，即将文本和图像数据结合起来发送给支持多模态的语言模型
# 核心概念
# ChatOpenAi：初始化一个支持多模态的聊天模型，例如gpt-40
# SystemMessage：定义模型的角色和行为
# HumanMessage：构造包含多模态内容的HumanMessage
# - type: "text"：用于包含文本内容
# - type: "image_url"：用于包含图像内容，其中image_url字段可以是一个公共访问的URL
#
#

import os
from typing import Any, Dict, List

import dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

dotenv.load_dotenv()

OPEN_API_URL = os.environ.get("OPEN_API_URL")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")


def run_multimodal_prompt():
    """
    演示使用多模态提示
    """
    image_url = "https://www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png"

    messages = [
        SystemMessage(content="你是一个图像分析助手，能够理解图像内容并回答相关问题"),
        HumanMessage(
            content=[
                {"type": "text", "text": "这张图片描绘了什么？请详细描述。"},
                {"type": "image_url", "image_url": image_url},
            ]
        ),
    ]
    if OPEN_API_KEY is None:
        raise ValueError("OPENAI_KEY_V4 is not set")
    llm = ChatOpenAI(
        base_url=OPEN_API_URL,
        api_key=SecretStr(OPEN_API_KEY),
        model="gpt-4o",
    )
    try:
        response = llm.invoke(messages)
        print(f"模型响应:{response.content}")
    except Exception as e:
        print(f"错误信息:{str(e)}")


if __name__ == "__main__":
    run_multimodal_prompt()
