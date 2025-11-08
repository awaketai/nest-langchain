# 将外部数据加载未document对象，然而这些文档通常太长，无法直接放入模型的上下文窗口
# 为了解决这个问题，我们需要将一个大的document分割吃呢共多个小的，语义相关的"块"(chunks)
# 这就是文本分割器的作用

# 为什么需要文本分割器
# 1.适应上下文窗口：这是最主要的原因，将长文档分割成小块，可以让我们在处理时只关注与用户问题最相关的部分
# 2.提升检索效率：在RAG流程中，我们通常会对这些小块进行嵌入并存入向量数据库。更小、更具语义焦点的文本块
# 能够提供更精确的检索结果。如果一个文本块太大，它可能包含太多不相关的主题，从而稀释了其向量表示的语义

# 文本分割的核心思想
# 一个好的文本分割策略应该力求在保持语义完整性的同时，将文本切分成合适的大小
# chunk_size和chunk_overlap
# 配置文本分割器时，有两个核心参数
# chunk_size:定义了每个文本块的最大尺寸，这个尺寸通常以字符数或token数来衡量
# chunk_oerlap:定义了相邻文本之间重叠的字符(或)token数，设置一个小的重叠(例如100-200个字符)
# 是一个非常好的实践，它有助于在两个块的边界处保持语义的连续性，避免一个完整的寓意单元(如一个长句子)
# 被硬生生切断在两个独立的块中
# LangChain的文本分割器
# 1.RecursiveCharacterTextSplitter:这是推荐的，最常用的分割器，它的“递归”之处在于它会尝试用一个分割列表
# (默认是["\n\n", "\n", " ", ""])来依次进行分割。它首先尝试用段落(\n\n)来分割，如果分割后的块仍然太大，它会接着尝试用
# 换行符(\n)来分割这个大块，一次类推
# 2.CharacterTextSplitter:一个更简单的分割器，它只使用你指定的一个分隔符(例如\n\n)来进行分割
# 3.TokenTextSplitter:直接根据token的数量来分割文本。这对于精确控制输入到模型的token数量非常有用，但需要一个分词器(tokenizer)
# 4.特定语言的分割器：LangChain还为Python,JaavaScript,Markdown等特定语言提供了专门的分割器，它们能够理解这些语言的语法结构
# 并据此进行更智能的分割
#
# 使用RecursiveCharacterTextSplitter进行文档分割

import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def recursive_character_text_splitter():
    file_path = os.path.join(os.path.dirname(__file__), "./static/long_text_sample.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    original_doc = documents[0]
    print(f"原始文档字符数:{len(original_doc.page_content)} \n")

    # 创建RecursiveCharacterTextSplitter实例
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True,  # 在元数据中添加每个块在原文中的起始位置
    )
    # 分割文档
    split_docs = text_splitter.split_documents(documents)
    print(f"原始文档数量:{len(documents)} 分割后文档数量:{len(split_docs)}\n")

    print("-----分割后的块-----\n")
    print("第一个块\n")
    print(split_docs[0].page_content)
    print(f"元数据:{split_docs[0].metadata}")
    print(f"长度:{len(split_docs[0].page_content)}")

    print("第二个块\n")
    print(split_docs[1].page_content)
    print(f"元数据:{split_docs[1].metadata}")
    print(f"长度:{len(split_docs[1].page_content)}")

    print("第三个块\n")
    print(split_docs[2].page_content)
    print(f"元数据:{split_docs[2].metadata}")
    print(f"长度:{len(split_docs[2].page_content)}")


if __name__ == "__main__":
    recursive_character_text_splitter()
