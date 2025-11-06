#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from langchain.prompts import PromptTemplate,ChatPromptTemplate


# 提示是给大语言模型LLM的指令，一个好的提示是获得高质量、相关且安全回答的关键
# 提示模板是预定义的、可复用的、带有参数的"配方"，用于生成最终的提示

# 为什么使用提示模板
# 1.复用性Reusability:定义一次，随处使用
# 2.参数化Parameterization:模板可以包含变量，允许将用户的输入、从数据库检索到的上下文或者
# 其他动态信息插入到提示中
# 3.可读性与维护性Readability & Maintenance: 将复杂的提示逻辑从你的应用代码中分离出来，使得提示
# 本身更容易阅读、修改和版本控制
# 4.安全性：通过固定提示的结构，可以减少用户通过”提示注入“来改变模型行为的风险

# LangChain中的提示模板
# LangChain提供了多种类型的模板，最核心的两种
# 1.PromptTemplate
# 这是最基础的模板，用于生成一个简单的字符串提示，它接收一个模板字符串和一组输入变量
#

# prompt_template = PromptTemplate(
#     "给我讲一个关于{subject}的笑话"
# )
# prompt_string = prompt_template.format(subject="程序员")

# 2.ChatPromptTemplate
# 这是更常用、更强大的模板，用于生成一个消息列表，专门为ChatModel设计
# ChatPromptTemplate 通常由一个或多个消息提示模板(MessagePromptTemplate)组成
# 每个都对应一种消息类型(System，Human,AI)
# ChatPromptTemplate的输出是一个消息列表，可以直接传递给ChatModel


# chat_template = ChatPromptTemplate.from_messages([
#     ("system","你是一个专业的 {role}"),
#     ('human','你好，我的名字是 {name}'),
# ])
# prompt_messages = chat_template.format_messages(
#     role="翻译家",
#     name='小明'
# )

# 组合与高级用法
# LangChain的提示模板是可以组合的，可以将多个模板拼接在一起
# 或者使用PipelinePromptTemplate来构建更复杂的、分阶段的提示生成流程

# 1.两种模板构建方式
# 1.1 直接传入(role,template)元组列表，快速定义系统与用户消息模板，并查看input_variables了解需要提供的参数
# 1.2 使用SystemMessagePromptTemplate与HumanMessagePromptTemplate等子类，获得更强的可组合型与类型提示支持
# 2.消息格式化
# 调用format_message()传入不同的style、product等参数即可生成结构化的SystemMessage/HumanMessage列表，输出可直接送入聊天模型
# 3.接入LCEL链：当启用在线模式时，示例构建chat_template|ChatOpenAI|StrOutputParser的链式调用
# 输入字典会先填充模板，再由ChatOpenAI生成文本、最后调用StrOutputParser提取纯字符串结果

from llm_client_asher.llm_config import Config
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

cfg = Config()

def main():
    """
    本例演示了 `ChatPromptTemplate`的几种常见用法：
    1.直接使用消息类型和模板字符串的列表
    2.使用 `MessagePromptTemplate` 子类构建更结构化的提示
    3.将提示与 `ChatModel`组合成LCEL链完成一次调用
    :return:
    """
    # 1.使用元组列表创建ChatPromptTemplate
    print('----1.使用元组列表创建----\n')
    chat_template_1 = ChatPromptTemplate.from_messages([
        ('system',"你是一个专业的助手，擅长撰写 {style} 风格的文章。"),
        ('human','为产品 {product} 写一句宣传语')
    ])

    print(f'聊天模板1需要输入的变量:{chat_template_1.input_variables}\n')
    formatted_messages_1 = chat_template_1.format_messages(
        style='幽默',
        product='自动洗碗机',
    )
    print('格式化后的消息列表-1\n')
    for msg in formatted_messages_1:
        print(f'----类型:{type(msg).__name__} 内容:{msg.content}')

    print('-' * 30)

    # 2.使用 MessagePromptTemplate对象列表创建
    print('\n----2.使用MessagePromptTemplate对象穿件----\n')
    system_prompt = SystemMessagePromptTemplate.from_template(
        '你是一个专业的助手，擅长写 {style} 风格的文案。'
    )
    human_prompt = HumanMessagePromptTemplate.from_template(
        '为产品 {product} 写一句宣传语'
    )

    chat_template_2 = ChatPromptTemplate.from_messages([
        system_prompt,
        human_prompt,
    ])
    formatted_messages_2 =  chat_template_2.format_messages(
        style='正式',
        product='高端手表',
    )
    print(f'格式化后的消息列表-2:\n')
    for msg in formatted_messages_2:
        print(f'-类型 {type(msg).__name__}: {msg.content}')
    print('-' * 30)

    # 3.在LCEL链中使用
    print(f'---3.在LCEL链中调用---\n')
    try:
        chat_model = ChatOpenAI(
            base_url=cfg.OPENAI_BASE_URL,
            api_key=cfg.OPENAI_KEY_V4,
        )
        chain = chat_template_2 | chat_model | StrOutputParser()
        chain_inputs = {
            'style':'正式',
            'product':'高端手表'
        }
        print(f'链的输入:{chain_inputs} \n')
        response_text = chain.invoke(chain_inputs)
        print(f'---response: {response_text} \n')
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()




