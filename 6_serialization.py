#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from llm_client_asher.llm_config import Config
from llm_client_asher.llm_factory import LLMFactory,LLMType
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.load import load,dumps,loads

cfg = Config()
os.environ['OPENAI_API_KEY'] = cfg.OPENAI_KEY_V4
os.environ['ANTHROPIC_API_KEY'] = cfg.CLAUDE_API_KEY_V4

def serialize():
    prompt = ChatPromptTemplate.from_template("写一首关于{topic}的诗")
    model = LLMFactory(
        client_type=LLMType.ANTHROPIC_AI,
        max_tokens=1000
    ).get_llm()
    print("--model:",model)
    parser = StrOutputParser()
    chain = prompt | model | parser
    print("----原始的LCEL链----")
    # 打印链结构
    try:
        for i,step in enumerate(chain.get_graph().nodes.values()):
            print(f"Step {i + 1}: {step.name}")
    except Exception as e:
        print(f"无法直接打印图结构，但链已创建:{e}")
    print("-" *30)
    # 序列化并保存到文件
    file_path = "my_lcel_chain.json"
    # 使用dumps将链对象序列化为json字符串
    chain_json = dumps(chain,pretty=True)
    with open(file_path,"w",encoding="utf-8") as f:
        f.write(chain_json)

    print(f"\nLCEL 链已序列化并保存到:{file_path}")
    print("-" * 30)
    # 查看文件内容
    with open(file_path,"r",encoding="utf-8") as f:
        chain_json = f.read()
        loaded_chain = loads(chain_json)
    print("\n---从文件加载的 LCEL 链---")
    try:
        for i,step in enumerate(loaded_chain.get_graph().nodes.values()):
            print(f"Step {i + 1}: {step.name}")

    except Exception as e:
        print(f"无法直接打印图结果，但链已从文件加载:{e}")
    # 验证加载的链是否能正常工作
    response = loaded_chain.invoke({"topic":"月光"})
    print(response)

if __name__ == "__main__":
    serialize()