# Runnable Interface：所有可执行组件的基础接口
# 包括LLM、Prompt模板、输出解析器、Retriever等
# 核心方法：
# invoke/ainvoke： 将单个输入转换为输出
# batch/abatch：批量处理输入，转换为输出
# stream/astream：流式处理输入，逐个生成输出
# astream_log：实时输出流式结果，同时输出中间过程


from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import RunnableLike

# 来自官方示例
# A RunnableSequence constructed using the `|` operator
sequence = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
resp = sequence.invoke(1)
print(resp)
resp = sequence.batch([1, 2, 3])
print(resp)
# A sequence that contains a RunnableParallel constructed using a dict literal
sequence = RunnableLambda(lambda x: x + 1) | {
    "mul_2": RunnableLambda(lambda x: x * 2),
    "mul_5": RunnableLambda(lambda x: x * 5),
}
resp = sequence.invoke(1)
print(resp)
