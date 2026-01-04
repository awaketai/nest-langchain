# RAG Retrieval Augmented Generation
# 数据源 -> Text Split -> Embedding -> Vector Store -> Retriever -> Prompt -> LLM


import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

load_dotenv()

OPEN_API_URL = os.environ.get("OPEN_API_URL")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")

# 1.文档加载
loader = TextLoader("./static/doc1.txt")
docs = loader.load()

# 2.文本切分
# chunk_size 约等于 模型上下文 1/5 ~ 1/10
# overlap 是为了语义连续性，不是越大越好

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)

# 3.Embedding
# Embedding 是相似度工具，不是理解工具
if OPEN_API_KEY is None:
    raise ValueError("OPENAI_KEY_V4 is not set")
embedder = OpenAIEmbeddings(
    base_url=OPEN_API_URL,
    api_key=SecretStr(OPEN_API_KEY),
    model="text-embedding-3-large",
)
# 4.向量库
vecstorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embedder,
)
# 5.Retriever
# 找什么比怎么答更重要
retriever = vecstorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)
# 6.Prompt
prompt = ChatPromptTemplate.from_template(
    """
    你是一个严谨的技术助手。
    只允许根据【上下文】回答问题。
    如果上下文没有答案，请直接说“不知道”。

    【上下文】
    {question}
    """
)

model = ChatOpenAI(
    base_url=OPEN_API_URL,
    api_key=SecretStr(OPEN_API_KEY),
    temperature=0,
)

chain = (
    {
        "context": retriever,
        "question": lambda x: x,
    }
    | prompt
    | model
    | StrOutputParser()
)

answer = chain.invoke("AI的分支是什么")
print(answer)
