# 文档被加载和分割后，这些文本仍然是字符串，计算机无法直接理解它们的含义
# 为了让机器能够比较文档的相似性(例如，判断哪个文本块与用户的问题最相关)
# 需要一种方法将文本转换为数字表示，这就是嵌入模型的作用

# 什么是嵌入模型
# 嵌入时一个将离散的变量(如单词、句子或整个文档)转换为连续向量空间中的一个向量(一长串浮点数)的过程
# 这个过程的关键在于：嵌入模型经过训练，能够将语义上相似的文本映射到向量空间中相近的点
# “小猫”和“猫咪”的向量会非常接近
# “今天天气怎么样”和“查询当前气候状况”的向量会很接近
# 通过计算两个文本向量之间的“距离”，通常使用余弦相似度，我们就可以量化它们在语义上的相似程度
# 这就是所有现代语义搜索和RAG应用的数学基础

# 当内部文档存储到向量数据库之后，就需要根据问题和任务来提取最相关的信息，信息提取的基本方式就是
# 把问题转换为向量，然后去和向量数据库中的各个向量进行比较，提取最接近的信息
# 向量之间的比较通常基于向量的距离或者相似度，在高维空间中，常用的向量距离或者相似度计算方法有欧氏距离和余弦相似度
# 欧氏距离：这是最直接的距离度量方式，就像在二维平面上测量亮点之间的直线距离那样
# 余弦相似度：在文本处理中，一个词的向量可能会因为文本长度的不同，而在大小上有很大的差距，但方向更能反映其语义
# 余弦相似度就是度量向量之间的相似性，它的值范围在-1到1之间，值越接近1，表示两个向量的方向越相似

# LangChain中的嵌入模型
# LangChain提供了一个标准的Embeddings类接口，使得与各种嵌入模型提供商，如OpenAI、Hugging Face等进行集成变得简单
# 这个接口主要有两个方法
# 1.embed_documents(self,texts: List[str]) -> List[List[float]]
# 输入：一个文本字符串列表(通常是文档块的内容)
# 输出：一个向量列表，每个向量对应一个输入文本
# 用途：用于在构建向量数据库时，批量的为所有文档块创建嵌入
# 2.embed_query(self,text:str) -> List[float]
# 输入：一个单独的文本字符串(通常是用户的查询)
# 输出：一个代表该查询的向量
# 用途：用于在进行相似性搜索时，为用户的实时查询创建嵌入
#
# 为什么需要两个方法
# 一些模型提供商针对“存储用的文档”和“查询用的问题”训练了不同的嵌入模型，embed_documents和embed_query的分离设计正式为了适应这个
# 情况，以达到最佳的检索效果
# 常见的嵌入模型
# OpenAIEmbeddings:使用OpenAi的嵌入模型，如text-mbedding-3-small，性能优异，是商业应用中最常见的选择之一
# HuggingFaceEmbeddings:允许你从Hugging Face HUb加载和运行各种开源的嵌入模型，这对于希望在本地运行模型或有特性语言需求(如中文)的场景非常有用
# GoogleGenerativeAIEmbeddings:使用Google的嵌入模型，如text-embedding-gecko，性能优异，是商业应用中最常见的选择之一
#


# from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
#
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


def embed_doc():
    # 1.初始化模型，指定在CPU上运行，并规范化嵌入向量
    embedder = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 示例中文文档
    documents = [
        "今天天气真好，万里无云。",
        "我喜欢吃冰淇淋，尤其是草莓味的。",
        "大型语言模型是深度学习的一个重要分支。",
        "如何才能高效地学习编程？",
    ]
    # 使用embed_documents和embed_query分别为文档和查询创建嵌入向量，并计算相似度
    #
    doc_vecs = embedder.embed_documents(documents)

    print(
        f"Embedded {len(doc_vecs)} documents.Vector dim(first):{len(doc_vecs[0]) if doc_vecs else 0}"
    )

    query = "学习AI需要哪些知识？"
    q_vec = embedder.embed_query(query)

    # 计算相似度(点积)
    # 将文档嵌入向量转换成NumPy数组，方便进行矩阵运算
    doc_np = np.array(doc_vecs)
    # 将查询结果转换为NumPy数组
    q_np = np.array(q_vec)
    # 计算每个文档与查询向量之间的点积
    sims = doc_np.dot(q_np)

    # 打印相似度结果
    print("Similarity scores:\n")
    for i, doc in enumerate(documents):
        print(f"Document {i + 1}: '{doc}' \n 相似度: {sims[i]:.4f}\n")
    # 超出最相似的文档
    most_similar_idx = np.argmax(sims)
    print(f"最相关的文档:{documents[most_similar_idx]}")
    print(f"相似度得分:{sims[most_similar_idx]:.4f}")


if __name__ == "__main__":
    embed_doc()
