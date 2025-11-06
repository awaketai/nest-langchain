#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from langchain.chains.constitutional_ai.prompts import examples
from langchain_community.docstore import InMemoryDocstore
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
# 提示工程中，少样本学习(Few-shot Learning)时一种强大的技术
# 指的是在向模型提出实际问题之前，在提示中提供几个具体的：问题-答案示例
# 这些示例 可以引导模型更好地理解任务要求，并以我们期望的格式或风格生成回答
# 少样本提示的问题
# 直接在提示中硬编码的问题是，如果示例库非常大，我们不可能全部放进提示中，
# 会超出模型的上下文窗口限制，并且成本高昂
# 我们需要一种方法，能够根据用户的输入，从庞大的示例库中动态的选择一小部分最相关的示例
# 解决方案：示例选择器
# LangChain的ExampleSelector就是为此而生的，它是一个能够根据特定策略从一组候选项中
# 选择一部分示例的组件
# 常见的示例选择器类型：
# 1.LengthBasedExampleSelector(基于长度)
# 策略：根据示例格式化后的总长度来选择示例，它会不断添加示例，直到总长度接近设定的最大值max_length
# 优点：简单、有效，可以确保提示不会超出上下文限制
# 缺点：选择的示例与当前输入的相关性无关，只是简单的按顺序选取

# 2.SemanticSimilarityExampleSelector(基于语义相似度)
# 策略：这是最常用也是最强大的选择器，它首先将所有示例通过嵌入模型(Embedding Model)转换为向量，
# 并存储在向量数据库中，当新输入来时，它会计算输入与所有示例之间的语义相似度，并选择最相似的k个示例
# 优点：能够为当前输入挑选出最相关、最有帮助的示例，极大的提高了少样本学习的有效性
# 缺点：设置相对复杂，需要一个嵌入模型和一个向量数据库

# 3.MaxMarginalRelevanceExampleSelector(最大边际相关性)
# 策略：这是SemanticSimilarityExampleSelector的一个变种，它不仅选择与输入相似的示例，还能同时确保
# 示例之间的多样性，避免选出的示例过于同质化
# 优点：可以在提供相关性的同时，给模型更多样化的视角

from langchain_core.example_selectors.length_based import LengthBasedExampleSelector
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.vectorstores import VectorStore

examples = [
    {"input": "开心", "output": "难过"},
    {"input": "高", "output": "矮"},
    {"input": "快", "output": "慢"},
    {"input": "一个非常非常长的输入字符串，这会占用很多空间", "output": "一个同样非常非常长的输出字符串"},
    {"input": "白天", "output": "黑夜"},
]

def length_based():

    prompt = PromptTemplate.from_template("Input: {input}\nOutput: {output}")
    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=prompt,
        max_length=25,
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=prompt,
        prefix="请根据下面的示例，给出输入词的反义词。",
        suffix='输入: {user_input}\n 输出：',
        input_variables=['user_input'],
    )

    print(few_shot_prompt.format(user_input="开心"))

import faiss
# from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def semantic_similarity():
    """
    1.使用HuggingFaceEmbeddings构建嵌入模型
    2.通过resolve_faiss()动态导入FAISS向量检索器，确实依赖时会给出安装提示
    3.SemanticSimilarityExampleSelector.from_examples会把提示写入向量库，k=2表示返回两个最相似的示例
    4.与上一示例类似，FewShotPromptTemplate在格式化时自动注入与用户输入最接近的提示
    :return:
    """
    embeddings = HuggingFaceEmbeddings(
        model_name='mixedbread-ai/mxbai-embed-large-v1',
    )
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=examples,
        embeddings=embeddings,
        vectorstore_cls=FAISS,
        k=2,
    )
    prompt = PromptTemplate.from_template("Input: {input}\nOutput: {output}")
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=prompt,
        prefix="请根据下面的示例，为输入生成一个富有想象力的场景描述。",
        suffix='输入: {user_input}\n 输出：',
        input_variables=['user_input'],
    )
    final_prompt = few_shot_prompt.format(user_input='一头孤独的狼')
    print(final_prompt)

if __name__ == '__main__':
    semantic_similarity()
