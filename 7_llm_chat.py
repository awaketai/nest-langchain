#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from langchain_openai import OpenAI
from llm_client_asher.llm_config import Config
from pydantic import SecretStr

# 在LangChain中，与大语言模型交互式通过两种核心抽象实现
# LLM和ChatModel
# 1.LLM(文本补全模型)，相对简单的接口，遵循文本输入、文本输出的格式
# 输入：一个字符串 输出：一个字符串
# 这个接口非常适合于那些不涉及对话历史的、直接的文本补全任务

cfg = Config()
os.environ['OPENAI_API_KEY'] = cfg.OPENAI_KEY_V4

def llm_model():
    llm = OpenAI(
        api_key=SecretStr(cfg.OPENAI_KEY_V4),
        base_url=cfg.OPENAI_BASE_URL,
    )
    response = llm.invoke("从前有座山,")
    print(response)

# ChatModel是一个更强大、更灵活的接口，它围绕'聊天消息'的概念进行设计
# 输入：一个ChatMessage对象的列表
# 输出：一个AIMessage对象
# ChatMessage有几种不同的类型
# 1.SystemMessage:系统消息，用于设定AI的角色、个性和行为准则，通常放在消息列表的开头
# 2.HumanMessage:用户消息，代表用户的输入
# 3.AIMessage:AI消息，代表模型的回复，这可以用来在对话历史中展示模型之前的发言

# 为什么使用ChatModel
# 1.结构化对话：消息里列表的结构天然支持多轮对话和角色扮演
# 2.功能更强：现代的、最强大的模型都是通过聊天接口暴漏其全部功能
# 3.更优的性能：许多模型提供商针对聊天模式进行了专门的优化


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage

def chat_model():
    chat = ChatOpenAI(
        api_key=SecretStr(cfg.OPENAI_KEY_V4),
        base_url=cfg.OPENAI_BASE_URL,
    )

    messages = [
        SystemMessage(content='你是一个乐于助人的助手。'),
        HumanMessage(content='你好吗'),
    ]
    response = chat.invoke(messages)
    print(response)

if __name__ == '__main__':
    chat_model()