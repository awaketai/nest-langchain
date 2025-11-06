#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from llm_client_asher.llm_config import Config
from langchain_openai import ChatOpenAI
from llm_client_asher.llm_factory import LLMFactory
from llm_client_asher.llm_factory import LLMType

cfg = Config()

# 大语言模型的输出本质上是字符串，输出解释器就是负责将LLM的原始文本输出转换为结构化数据的组件
# 它们是连接语言模型和应用程序逻辑的桥梁
# 输出解析器的工作流程
# 1.提供格式化指令：大多数解析器都有一个get_format_instructions()方法，这个方法会生成一段文本
# 详细描述了模型应该如何格式化其输出，必须将这段指令包含在给模型的提示中，以引导模型生成符合要求的文本
# 2.解析输出：当模型返回一个字符串后，解析器的parse()方法会被调用，这个方法负责将该字符串解析为期望
# 的结构化数据格式，如果解析失败，会抛出一个OutputParserException
# LangChain提供的输出解析器
# 1.StrOutputParser:最基础的解析器，也是LCEL链中的默认解析器，它不做任何转换，只是简单的将模型的输出作为字符串返回
# 2.JsonOutputParser:将模型输出的JSON字符串解析为Python字典
# 3.PydanticOutputParser:这是功能最强大的解析器之一。它允许你定义一个Pydantic模型，然后将模型的输出直接解析为
# 该模型的实例，这不仅提供了结构化的数据，还附带了Pydantic提供的所有数据验证和类型提示功能，极大的提升了代码的健壮性
# 4.CommaSeparatedListOutputParser:将一个逗号分割的字符串，解析为一个Python列表
# 5.DatetimeOutputParser：解析与日期时间相关的文本

# 解析器是Runnable的，因此可以自然的作为LCEL链的最后一个环节
# chain = prompt | model | output_parser

def main():
    """
    使用JsonOutputParser解析器，指示模型输出JSON字符串，并自动将其解析为Python字典
    :return:
    """
    # 1.定义期望的JSON结构
    class Joke(BaseModel):
        setup: str = Field(description='笑话的铺垫部分')
        punchiline: str = Field(description="笑话的点睛之笔")


    # 2.创建一个JsonOutputParser实例
    # 可以传入pydantic_object来帮助生成格式化指令
    parser = JsonOutputParser(pydantic_object=Joke)

    # 3.创建提示模板，并包含格式化指令
    format_instructions = parser.get_format_instructions()
    prompt_template_str = """
    根据用户的问题，生成一个笑话。
    {format_instructions}
    用户问题：{query}
    """
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=['query'],
        partial_variables={'format_instructions': format_instructions},
    )

    print("---格式化指令:\n")
    print(format_instructions)
    print("-" * 30)

    # model = ChatOpenAI(
    #     base_url=cfg.OPENAI_BASE_URL,
    #     api_key=cfg.OPENAI_KEY_V4,
    # )
    model = LLMFactory(client_type=LLMType.ANTHROPIC_AI).get_llm()

    chain = prompt | model | parser
    query = "给我讲一个关于程序员的笑话"
    try:
        result = chain.invoke({'query':query})
        print(f'输出类型:{type(result) } \n')
        print("输出内容\n")
        from pprint import pprint
        pprint(result)

        print(f"笑话的铺垫:{result.get('setup')}")
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()