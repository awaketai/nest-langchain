#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 数据连接
# 在构建能够回答关于特定只是的应用时，第一步总是将这些外部数据加载到应用程序中
# 文档加载器(Document Loaders)就是负责这个任务的组件，它们的主要工作是从各种来源(如文本文件、PDF、网页、数据库等)
# 加载数据，并将其统一转换为LangChain的标准Document格式

# Document对象
# Document对象是LangChain中处理文本块的标准方式。它是一个包含两部分核心信息的数据类
# 1.page_content(字符串)：这是文档的主要内容，一个文本字符串
# 2.metadata(字典)：这是一个包含关于文档元数据的字典。这些元数据对于后续的过滤、引用和分析非常重要
# 常见的元数据包括：
# source:文档的来源(例如：文件名、URL、数据库表名)
# page:文档在原始来源中的页码(例如：PDF的页码)
# row:文档在原始来源中的行号:(例如：CSV文件的行号)
# 为什么需要文档加载器？
# 文档加载器维尼处理了从各种复杂源众提取文本的繁琐工作，LangChain已经构建的文档加载器
# 文件：文本文件.txt csv json markdown pdf word等
# 网页：加载并解析HTML页面的内容
# 数据库：从SQL或NoSQL数据库中加载数据
# 协作工具：Notion,GoogleDrive,Confluence等
# API:通过API断点获取数据
# 使用文档加载器，只需要提供源的路径或URL，然后调用.load()方法，就能得到一个Docuemnt对象列表，可以直接在
# LangChain的其他组件(如文本分割器、嵌入模型)中使用
#
# 大多数加载器都提供两种方法：
# 1.load() 一次性将所有数据加载到内存中，返回一个List[Document]，对于大型数据集，这会消耗大量内存
# 2.lazy_load():返回一个迭代器，可以在需要时逐个加载文档，对于处理大型文件或数据集非常高效


# 使用TextLoader读取本地文本文件
import asyncio
import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    pdf,
)
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from llm_client_asher.llm_config import Config
from pydantic import SecretStr

cfg = Config()


def interact_with_llm(doc):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名中文助手，请以简洁的语言概括用户提供的文档要点",
            ),
            ("human", "请阅读一下文档内容并给出三句话的概要:\n\n{document}"),
        ]
    )

    if cfg.OPENAI_KEY_V4 is None:
        raise ValueError("OPENAI_KEY_V4 is not set")
    llm = ChatOpenAI(
        base_url=cfg.OPENAI_BASE_URL,
        api_key=SecretStr(cfg.OPENAI_KEY_V4),
        temperature=0,
    )
    chain = prompt | llm | StrOutputParser()
    try:
        summary = chain.invoke({"document": doc.page_content})
    except Exception as e:
        print(f"Error occurred while invoking the chain: {e}")
        raise ValueError(f"Error occurred while invoking the chain: {e}")

    print(f"----3.生成摘要:{summary.strip()}\n")

    follow_up_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名帮助回答问题的中文助手。"),
            ("human", "基于这份文档内容回答问题:{question}\n\n 文档:{document}"),
        ]
    )
    follow_up_chain = follow_up_prompt | llm | StrOutputParser()
    question = "这份文档建议的主要使用场景是什么？"
    try:
        answer = follow_up_chain.invoke(
            {"question": question, "document": doc.page_content}
        )
    except Exception as e:
        print(f"Error occurred while invoking the chain: {e}")
        raise ValueError(f"Error occurred while invoking the chain: {e}")

    print(f"----4.回答问题:{answer.strip()}\n")


def load_file_and_use_llm(loader: BaseLoader):
    docs = loader.load()
    if not docs:
        raise ValueError("No document loaded")

    doc = docs[0]
    from pprint import pprint

    print("文档元数据\n")
    pprint(doc.metadata)

    print("\n文档内容\n")
    print(doc.page_content)
    interact_with_llm(doc)


def get_file_path(file_name: str) -> str:
    return os.path.join(os.path.dirname(__file__), file_name)


def text_loader():
    """
    使用TextLoader读取本地文本文件
    """
    file_path = get_file_path("./static/example.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"----1.加载文件:{file_path} \n")
    loader = TextLoader(file_path, encoding="utf-8")
    load_file_and_use_llm(loader)


def pdf_loader():
    file_path = get_file_path("./static/example.pdf")
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    loader = PyPDFLoader(file_path)
    load_file_and_use_llm(loader)


async def web_loader():
    url = "https://python.langchain.com/docs/how_to/chatbots_memory/"
    loader = WebBaseLoader(web_paths=[url])
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    interact_with_llm(docs[0])


if __name__ == "__main__":
    # text_loader()
    # pdf_loader()
    asyncio.run(web_loader())
