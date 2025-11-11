# 将文档块转换为向量后，需要一个地方来存储这些向量，并能够高效的进行搜索
# 这就是“向量存储”，也就是向量数据

# 什么是向量存储
# 向量存储时一个专门为存储和查询高维向量而设计的数据库
# 假设有数百万哥文档块，每个都对应一个1536维的向量，当用户提出一个问题时，需要将问题也转换为一个向量
# 然后再这数百万哥向量中找到与问题向量最接近的几个
# 如果对每个问题都进行一次穷举的距离计算，这个过程将会非常慢
#
# 向量存储使用专门的索引算法，来执行“近似最相邻”搜索，ANN搜索能够在牺牲极小精确度的情况下，实现比暴力搜索
# 块几个数量级的查询速度，这对于构建实时响应的RAG应用至关重要
#
# 核心工作流程
# 1.添加(Adding/Indexing)：将文档块(Document对象)和他们的嵌入向量一起添加到向量存储中。大多数LangChain的向量存储
# 集成都提供了一个便利的from_documents方法，可以一步完成文档的嵌入和添加
#
# 2.查询(Querying):当用户提问时，首先用嵌入模型将问题转换为一个“查询向量”
# 3.搜索(Searching):将查询向量传递给向量存储的搜索接口，向量存储会利用其索引快速返回与查询向量最相似的k个文档
#
# LangChain中的向量存储
# LangChain提供了多种向量存储的集成，并提供了一个统一的VectorStore接口，这意味着可以轻松的在不同的向量存储后端之间切换
# 而无需修改大量代码
# 常见的向量存储
# FAISS: Facebook AI Similarity Search.一个非常高效的开源向量搜索库，它可以在内存中运行，非常适合快速原型开发和中小型应用
# Chroma:一个开源的，为AI应用设计的向量数据库，它可以作为本地服务运行，也可以在内存中运行
#
# LangChain的VectorStore接口提供了几个关键的搜索方法：
# similarity_search(query,k=4,**kwargs):这是最常用的方法，它接收一个查询字符串，自动将其嵌入，然后返回k个最相近的Document对象
# similarity_search_by_vector(embedding,k=4,**kwargs):与上面类似，但它接收的是一个已经嵌入好的向量查询
# max_marginal_relevance_search():一种更高级的搜索方法，它在选择文档时不仅考虑与查询的相似度，还考虑文档之间的多样性，以避免返回聂荣过于
# 同质化的结果
#
#
#
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# 分数解读
# 1.分数越大，表示文档与查询越相似(与余弦距离相反)
# 2.深圳的文档得分最高(0.6012)，说明它与查询(中国那个城市科技最发达？)最相关
# 3.杭州的文档得分最低(0.3293)，说明它与查询问题最不相关
def vector_store_search():
    """
    展示在处理非英文内容时的最佳时间，通过使用专门的中文嵌入模型，我们可以获得比通用模型更好
    的搜索效果
    1.如何使用HuggingFace的中文嵌入模型(shibing624/text2vec-base-chinese)
    2.如何对中文文档向量化和存储
    3.如何执行基于中文的相似性搜索
    4.如何获取搜索结果的相似度分数
    """
    # 初始化中文嵌入模型
    embedder = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # 规范化嵌入向量
    )
    # 准备一些中文文档
    documents = [
        Document(page_content="北京是中国的首都，有着悠久的历史文化。"),
        Document(page_content="上海是中国最大的经济中心，是一座现代化的国际大都市。"),
        Document(page_content="杭州以西湖闻名，是著名的旅游城市。"),
        Document(page_content="深圳是中国改革开放的窗口，高科技产业发达。"),
        Document(page_content="吐鲁番以其独特的火焰山和葡萄沟著称。"),
    ]
    # 创建向量存储
    # 使用DistanceStrategy.MAX_INNER_PRODUCT策略，使员工向量的内积作为相似度度量，其特点是：
    # 分数范围：在enbedder中设置了normalize_embeddings=True，向量都被归一化了，所以内积的值范围在[-1,1]之间
    #
    db = FAISS.from_documents(
        documents=documents,
        embedding=embedder,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    # 执行相似性搜索
    query = "中国那个城市科技最发达？"
    docs = db.similarity_search(query)
    print("搜索结果:\n")
    # enumerate(iterable,start=0)
    # 参数1：要被遍历的可迭代对象 参数2：起始索引
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content}")

    # 执行相似性搜索并返回相似度分数
    docs_and_scores = db.similarity_search_with_score(query)
    print("-" * 30)
    for i, (doc, score) in enumerate(docs_and_scores, 1):
        print(f"{i}. {doc.page_content} (相似度分数：{score:.4f})\n")


if __name__ == "__main__":
    vector_store_search()
