# 回调函数：函数A作为参数传递给另一个函数B，然后在函数B内部执行函数A，当函数B完成某些操作后，会调用函数A
# 异步操作使用回调函数示例
#
# 以下部分代码来自极客时间 黄佳老师LangChain实战课
import asyncio

from langchain_core.callbacks import AsyncCallbackHandler


async def compute(x, y, callback):
    print("Starting compute")
    await asyncio.sleep(0.5)  # 模拟异步操作
    result = x + y
    callback(result)
    print("Finished compute...")


def print_result(value):
    print(f"The result is: {value}")


async def another_task():
    print("Starting another task...")
    await asyncio.sleep(1)
    print("Finished another task...")


async def main():
    print("Main starts")
    task1 = asyncio.create_task(compute(3, 4, print_result))
    task2 = asyncio.create_task(another_task())

    await task1
    await task2
    print("Main ends")


# 上面的示例中，当调用asyncio.create_task(compute(3,4),print_result)，compute函数开始执行，当遇到 await asyncio.sleep(0.5) 时
# 它会暂停，并将控制权交换给事件循环，这时，时间循环可以选择开始执行another_task
#
# LangChain中的Callback处理器
# LangChain的callback机制允许在应用程序的不同阶段进行自定义操作，如日志记录、监控和数据流处理，这个机制通过Callbackhandler(回调处理器)
# 来实现
# 回调处理器是LangChain中实现CallbackHandler接口的对象，为每类可监控的事件提供一个方法。当该时间被触发时，CallbackManager会在这些处理器上调用
# 适当的方法
# BaseCallbackHandler是最基本的回调处理器，你可以继承它来创建自己的回调处理器。它包含了多种方法：
# - on_llm_start: 当LLM启动时
# - on_chat_model_start: 当聊天模型运行时
# - on_retriever_start 当检索启动时
if __name__ == "__main__":
    asyncio.run(main())
