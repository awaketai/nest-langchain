# 构建RAG:加载文档 -> 分割文本 -> 创建侵入 -> 向量存储
# 对于一次性操作，这个流程是可行的
# 但是当数据源变化时：文档被添加、更新、删除，每次数据源变化时，都需要高效的同步
# 变更到向量存储中，而不是每次都从头重建整个数据库
# 这就是LangChain的索引api要结局的问题
#
# 什么是索引API
# 索引API是一个高级工具，它将”加载、分割、嵌入、存储“的整个流程封装起来，并增加了状态管理和同步的功能
# 它的核心是langchain_community.indexes.index函数
# 1.幂等的处理文档：它或计算每个文档内容的哈希值，如果多次索引相同的内容，它只会处理一次，从而避免了重复劳动
# 2.高效的同步变更：它能够检测到哪些文档是新增的、哪些是被更新的，以及哪些是已经被删除的，并只对这些发生变化的文档进行操作
#
# 核心组件
# 索引API的魔力来自于RecordManager
# 作用：RecordManger负责追踪哪些文档已经被索引到了向量存储中，它通过一个键值存储(通常是一个sql数据库)来记录每个文档的”源id“和其内容的哈希值
# 工作流程：
# 1.当调用index时，它会首先从数据源加载所有当前的文档
# 2.对于每个加载的文档，它会去RecordManager中查询该文档的”源ID“
# 3.如果记录存在，它会比较新旧文档内容的哈希值，如果哈希值相同，说明文档内容未变，跳过，如果不同，则标记为更新
# 4.如果记录不存在，则标记为”新增“
# 5.index函数会批量的将所有新增和更新的文档写入向量存储
# 6.index会进行清理操作，将在RecordManager中存在但本次加载中未出现的文档从向量存储中删除
#
# inde函数参数
# docs_source:要索引的文档来源，可以是一个文档加载器，也可以是一个文档列表
# record_manager:用于追踪状态的记录管理器实例
# vector_store:目标向量存储实例
# cleanup:清理模式。可以是incremental(增量清理，只删除在源中消失的文档)，或full(全量清理，删除向量存储中所有不属于当前docs_source的文档)
# source_id_key:一指定了Document对象元数据中哪个键可以唯一的标识一份源文档，

import json
import os
import sqlite3
from datetime import datetime

from aiohttp.typedefs import Query
from langchain.indexes import SQLRecordManager, index
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentIndex:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_schema()

    def create_schema(self):
        """ "创建数据库schema"""
        with self.conn:
            self.conn.execute("""
             CREATE TABLE IF NOT EXISTS documents (
               source_id TEXT PRIMARY KEY,
               content TEXT,
               metadata TEXT,
               last_updated TIMESTAMP
             )
                  """)

    def update_document(self, source_id: str, content: str, metadata: str):
        """更新或插入文档"""
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO documents (source_id, content, metadata, last_updated)
                VALUES (?, ?, ?, ?)
                """,
                (source_id, content, metadata, datetime.now()),
            )

    def delete_document(self, source_id: str):
        """删除文档"""
        with self.conn:
            self.conn.execute("DELETE FROM documents WHERE source_id = ?", (source_id,))

    def get_document(self, source_id: str):
        """获取单个文档"""
        cursor = self.conn.execute(
            "SELECT source_id, content, metadata, last_updated FROM documents WHERE source_id = ?",
            (source_id,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "source_id": row[0],
                "content": row[1],
                "metadata": json.loads(row[2]) if row[2] else {},
                "last_updated": row[3],
            }

        return None

    def list_documents(self):
        """列出所有文档"""
        cursor = self.conn.execute(
            "SELECT source_id, content, metadata, last_updated FROM documents"
        )
        documents = []
        for row in cursor.fetchall():
            documents.append(
                {
                    "source_id": row[0],
                    "content": row[1],
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "last_updated": row[3],
                }
            )

        return documents

    def close(self):
        self.conn.close()


# 待补充文档索引管理器


def main():
    embedder = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # 规范化嵌入向量
    )
    vector_store = FAISS.from_documents(
        documents=[Document(page_content="初始化", metadata={"source": "init"})],
        embedding=embedder,
    )
    # 创建RecursiveCharacterTextSplitter实例
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True,  # 在元数据中添加每个块在原文中的起始位置
    )
    # 记录管理器
    record_manager = SQLRecordManager(
        namespace="demo", db_url="sqlite:///record_manager.db"
    )
    record_manager.create_schema()

    # 加载并处理文档
    doc_dir = os.path.join(os.path.dirname(__file__), "./static")
    # 索引多个文档
    docs = []
    for filename in ["doc1.txt", "doc2.txt"]:
        filepath = os.path.join(doc_dir, filename)
        loader = TextLoader(filepath, encoding="utf-8")
        docs.extend(loader.load())

    result = index(
        docs_source=docs,
        record_manager=record_manager,
        vector_store=vector_store,
        cleanup="incremental",
        source_id_key="source",
    )
    print(
        f"索引结果: 新增={result['num_added']}, 更新={result['num_updated']}, "
        f"跳过={result['num_skipped']}, 删除={result['num_deleted']} \n"
    )

    # 相似度搜索
    print("----相似度搜索----")
    query = "松软的"
    results = vector_store.similarity_search(query=query, k=2)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content} [来源: {doc.metadata['source']}] \n")

    # 5. 更新文档
    print("\n=== 更新文档 ===")
    with open("static/doc1.txt", "w", encoding="utf-8") as f:
        f.write("人工智能和深度学习正在改变世界。深度学习是AI的重要分支。")

    loader = TextLoader("static/doc1.txt", encoding="utf-8")
    updated_docs = loader.load()

    result = index(
        updated_docs,
        record_manager,
        vector_store,
        cleanup="incremental",
        source_id_key="source",
    )
    print(f"更新结果: 新增={result['num_added']}, 更新={result['num_updated']}")

    # 6. 搜索更新后的内容
    print("\n=== 搜索更新内容 ===")
    results = vector_store.similarity_search("深度学习", k=2)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content} [来源: {doc.metadata['source']}]")

    # 7. 删除文档（通过不包含在索引中实现）
    print("\n=== 删除文档 ===")
    # 只索引doc1，doc2会被自动清理（使用full模式）
    loader = TextLoader("static/doc1.txt", encoding="utf-8")
    remaining_docs = loader.load()

    result = index(
        remaining_docs,
        record_manager,
        vector_store,
        cleanup="full",  # full模式会删除未出现的文档
        source_id_key="source",
    )
    print(f"清理结果: 删除={result['num_deleted']}")

    # 8. 保存向量存储
    print("\n=== 保存向量存储 ===")
    vector_store.save_local("faiss_index")
    print("向量存储已保存到 faiss_index/")

    print("\n完成！")


if __name__ == "__main__":
    main()
