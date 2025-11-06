#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 在构建对话式应用中，我们需要一种能够处理多轮对话历史的提示结构
# ChatPromptTemplate正式为此设计的，它是与ChatModel配套使用的标准工具
# ChatPromptTemplate生成的是一个消息列表(List[BaseMessage])
# 每个消息都有自己的角色，如system,human,ai

# ChatPromptTemplate的核心组件
# 1.SystemMessagePromptTemplates:用于创建系统消息，通常为AI设定角色、背景和行为准则
# 2.HumanMessagePromptTemplate:用于创建用户消息，代表了对话中用户的输入
# 3.AIMessagePrmptTemplate:用于创建AI消息，可以用来在提示中提供一些少样本(few-shot)示例，引导AI遵循特定的回答格式

# MessagesPlaceholder: 处理对话历史的关键
# 在多轮对话中，我们不能预先知道对话会有多少轮，我们需要一个"占位符"来告诉模板，在这里插入一个动态的消息列表
# MessagePlaceholder允许在模板中指定一个变量，该变量在运行时会被一个消息列表所替换
# MessagePlaceholder是构建有记忆能力的聊天机器人的核心组件

from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import  HumanMessage,AIMessage
from langchain_core.output_parsers import StrOutputParser
from llm_client_asher.llm_config import Config
from langchain_openai import ChatOpenAI

cfg = Config()

prompt = ChatPromptTemplate.from_messages([
    ('system','你是一个乐于助人的助手'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{input}')
])

formatted_messages = prompt.format_messages(
    chat_history=[
        HumanMessage(content='你好'),
        AIMessage(content='你好，有什么可以帮你的吗？')
    ],
    input='LangChain是什么？'
)

model = ChatOpenAI(
    base_url=cfg.OPENAI_BASE_URL,
    api_key=cfg.OPENAI_KEY_V4,
)
# 方法一：直接调用
response = model.invoke(formatted_messages)
print(f"直接调用内容---:{response.content} \n")

# 方法二：使用LCEL
# StrOutputParser会自动从AIMessage中提取.content
chain = prompt  | model | StrOutputParser()
response = chain.invoke({
    "chat_history":[
        HumanMessage(content='你好'),
        AIMessage(content='你好，有什么可以帮你的吗？')
    ],
    "input":'LangChain是什么？'
})
print(f"使用LCEL调用:{response} \n")

# 创建ChatPromptTemplate最常见的是使用from_messages类方法
# 它可以接收多种格式的输入
# 1.(类型，内容)元组列表
# ChatPromptTemplate.from_messages(
# ('system','你是一个乐于助人的助手''),
# ('human','{question}')
#)
# 2.BaseMessagePromptTemplate对象列表
# from langchain.prompts import SystemMessagePromptTemplate,HumanMessagePromptTemplate
# ChatPromptTemplate.from_messages([
# SystemMessagePromptTemplate.from_template('...'),
# HumanMessagePromptTemplate.from_template(...'')
#])
# 3.BaseMessage对象列表
# from langchain_core.messages import SystemMessage,HumanMessage
# ChatPromptTemplate.from_messages([
# SystemMessage(content=''),
# HumanMessage(content='{question}')
#])
# 
# 消息类型Message Types
# 与ChatModel的所有交互都是通过消息(Messages)来进行的，消息是结构化的数据单元
# 它不仅包含内容,还带有一个明确的角色,告诉模型这段内容是谁说的
# LangChain将这些消息封装为不同的类，他们都继承自BaseMessage
# 1.HumanMessage 角色:human 代表对话中最终用户的输入，这是你向模型提问或给出指令的地方
# 2.AIMessage 角色:ai 代表AI模型的回应，当将AIMessage放入消息历史中再次发送给模型时，它就为
# 模型提供了之前的对话上下文，AIMessage还可以包含tool_calls,表示模型希望执行一个或多个工具
# 3.SystemMessage 角色:system 为AI设定高级指令、角色或上下文。SystemMessage通常作为消息列表的第一个元素
# 为整个对话定下基调
# 4.ToolMessage 角色:tool 用于将工具函数执行的结果返回给模型，当模型在前一步请求调用一个工具时(通过AIMessage的tool_calls属性)
# 你需要执行该工具，然后将结果包装在一个ToolMessage中发回给模型，以便它能生成最终的自然语言回答
# 每个ToolMessage都需要一个tool_call_id来与触发它的tool_calls中的具体调用相对应


