# 在LangChain中，链(Chain)是将多个组件(如语言模型、提示、工具)按顺序组合起来，以完成特定任务
# 的核心概念。通过将简单的组件链接在一起，可以构建出功能强大的复杂应用
#
# LCEL用于构建链的声明式方法，它使用管道符|将不同的组件连接起来，使得数据流进行流式处理
# 一个基础的链通常由一下三个部分组成
# 1.提示(Prompt):接收用户输入，并将其格式化为模型可以理解的提示词
# 2.模型(Model):接收格式化后的提示，并生成回应
# 3.输出解析器(OutputParser):将模型生成的回应转换为更已于使用的格式

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_client_asher.llm_config import Config
from pydantic import SecretStr

cfg = Config()

# TODO:待补充：
# 1.如何构建一个具有 记忆 (Memory) 功能的对话链。通过手动管理聊天历史，该链能够在多轮对话中记住上下文。
# 2.example_1_basic_agent.py & example_2_agent_with_memory.py: 这两个文件介绍了 **代理 (Agent)**，它是一种更高级、更动态的链。与固定步骤的链不同，代理能够使用大语言模型进行“思考”，并自主决定调用哪个工具（如搜索引擎）来完成任务。example_2 在 example_1 的基础上增加了记忆功能。


def simple_lcel_chain():
    """
    使用LCEL构建一个最简单的链，它将提示、模型输出解析器连接起来，用于生成一个笑话
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个幽默的喜剧演员，擅长讲笑话"),
            ("user", "给我讲一个关于{topic}的笑话"),
        ]
    )
    if cfg.OPENAI_KEY_V4 is None:
        raise ValueError("invalie open ai key")
    llm = ChatOpenAI(
        base_url=cfg.OPENAI_BASE_URL,
        api_key=SecretStr(cfg.OPENAI_KEY_V4),
    )
    output_parser = StrOutputParser()

    # 使用lcel的管道符
    chain = prompt | llm | output_parser
    res = chain.invoke({"topic": "程序员"})
    print(res)


def rag_chain():
    """
    构建一个RAG(Retrieval-Augmented Generation)链。该链首先从向量数据库中检索与用户问题相关的文档，
    然后将这些文档作为上下文，让模型生成更精准的回答
    """
    documents_text = """
    Python是一种高级编程语言,由Guido van Rossum于1991年创建。
    Python具有简洁易读的语法,广泛应用于Web开发、数据分析、人工智能等领域。
    LangChain是一个用于开发由语言模型驱动的应用程序的框架。
    RAG是检索增强生成的缩写,它结合了信息检索和文本生成。
    """
    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )
    texts = text_splitter.create_documents([documents_text])
    # 创建向量数据库
    if cfg.OPENAI_KEY_V4 is None:
        raise ValueError("invalie open ai key")
    embeddings = OpenAIEmbeddings(
        base_url=cfg.OPENAI_BASE_URL,
        api_key=SecretStr(cfg.OPENAI_KEY_V4),
    )
    vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 创建提示模板
    template = """
    基于以上上下文回答问题：
    上下文：{context}

    问题：{question}

    回答：
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(
        base_url=cfg.OPENAI_BASE_URL,
        api_key=SecretStr(cfg.OPENAI_KEY_V4),
        temperature=0,
    )
    output_parser = StrOutputParser()

    # 构建rag链
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    # 调用
    res = rag_chain.invoke("Python是什么时候创建的？")
    print(res)


if __name__ == "__main__":
    # simple_lcel_chain()
    rag_chain()
