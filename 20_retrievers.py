# 检索器
#
# 向量存储负责存储并按向量相似度搜索文档，然而，在LangChain的链式(LCEL)中，我们通常不与
# 向量存储直接交互，而是通过一个更高层次的抽象-”检索器“进行
#
# 检索器：一个通用的接口，接口一个字符串，返回一个相关的文档列表
# 核心方法：invoke(query:str) -> List[Document]
#
# 多重查询检索器：让LLM从不同角度重写用户的原始查询，然后并行执行这些查询并合并结果
#

# has problem when run
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from llm_client_asher.llm_config import Config
from pydantic import SecretStr

cfg = Config()


def multi_query_retriever():
    """
    多重查询检索器：
    让LLM从不同角度重写用户的原始查询，然后并行执行这些查询并合并结果
    """
    documents = [
        Document(page_content="苹果公司在2023年9月发布了iPhone 15系列"),
        Document(page_content="iPhone 15 Pro采用了钛金属边框，大大减轻了重量"),
        Document(page_content="新款iPhone支持USB-C接口，告别了Lightning接口"),
        Document(page_content="iPhone 15系列的主摄像头升级到了4800万像素"),
        Document(page_content="Pro机型支持可编程的操作按钮，取代了静音开关"),
    ]
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
    except Exception as e:
        raise ValueError(e)
    try:
        if cfg.OPENAI_KEY_V4 is None:
            raise ValueError("OPENAI_KEY_V4 is not set")
        llm = ChatOpenAI(
            base_url=cfg.OPENAI_BASE_URL,
            api_key=SecretStr(cfg.OPENAI_KEY_V4),
            temperature=0,
        )
        # 创建高级检索器，
        # 1.使用LLM将原始查询该写为多个不同角度的查询
        # 2.并行执行这些查询
        # 3.合并和去重结果
        retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(),
            llm=llm,
        )
    except Exception as e:
        raise ValueError(e)
    template = """
    根据以下文档回答问题：
    {context}

    问题:{question}
    回答:
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = {"context": retriever, "question": lambda x: x} | prompt | llm
    response = chain.invoke("新iPhone有哪些硬件变化？")
    print(response)


if __name__ == "__main__":
    multi_query_retriever()
