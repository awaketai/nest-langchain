# 回调函数：函数A作为参数传递给另一个函数B，然后在函数B内部执行函数A，当函数B完成某些操作后，会调用函数A
# 异步操作使用回调函数示例
#
# 以下部分代码来自极客时间 黄佳老师LangChain实战课
#
import asyncio
import os
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from dotenv import load_dotenv
from loguru import logger

# import GenerationChunk,ChatGenerationChunk,UUID
#

load_dotenv()

OPEN_API_URL = os.environ.get("OPEN_API_URL")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")

from langchain.chains import LLMChain
from langchain.schema import BaseMessage, HumanMessage, LLMResult
from langchain_core.callbacks import AsyncCallbackHandler, FileCallbackHandler
from langchain_core.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


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


async def main1():
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
#
# LangChain内置回调处理器
# StdOutCallbackHandler: 会将所有事件记录到标准输出，
# FileCallbackHandler: 将所有日志记录到一个指定的文件中
#
# 在组件中使用回调处理器
# 在LangChain的各个组件，如Chains/Models/Tools/Agents等，都提供了两种类型的回调设置方法：
# 构造函数回调和请求回调，可以在初始化LangChain时将回调处理器传入，或者在单独的请求中使用回调
# 例如：当想要在整个链的所有请求中进行日志记录时，可以在初始化时传入处理器
# 当只想在某个特定请求中使用回调时，可以在请求时传入
#
#
#
#
log_file = "output.log"


def log_callback():
    logger.add(log_file, colorize=True, enqueue=True)
    handler = FileCallbackHandler(log_file)
    if OPEN_API_KEY is None:
        raise ValueError("OPENAI_KEY_V4 is not set")
    llm = ChatOpenAI(
        base_url=OPEN_API_URL,
        api_key=SecretStr(OPEN_API_KEY),
        model="gpt-3.5-turbo",
        temperature=0,
    )
    prompt = PromptTemplate.from_template("1 + {number} =")
    chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler], verbose=True)
    answer = chain.invoke({"number": 2})
    print(answer)
    logger.info(answer)


# 通过BaseCallbackHandler和AsyncCallbackHandler实现自己的回调处理器
#
# 创建同步回调处理器
class MyFlowerShopsyncHandler(BaseCallbackHandler):
    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"获取花卉数据:token:{token}")


# 创建异步回调处理器
class MyFlowerShopAsyncHandler(AsyncCallbackHandler):
    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        print("正在获取花卉数据")
        await asyncio.sleep(0.5)
        print("花卉数据获取完毕...")

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        print("整理花卉建议")
        await asyncio.sleep(0.5)
        print("祝你今天愉快！")

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print("模型开始调用")


# 异步函数
async def main():
    if OPEN_API_KEY is None:
        raise ValueError("OPENAI_KEY_V4 is not set")
    llm = ChatOpenAI(
        base_url=OPEN_API_URL,
        api_key=SecretStr(OPEN_API_KEY),
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=100,
        streaming=True,
        callbacks=[MyFlowerShopsyncHandler(), MyFlowerShopAsyncHandler()],
    )

    # 异步生成聊天回复
    response = llm.generate(
        [[HumanMessage(content="哪种花卉最适合过生日，只简单说3种，不超过50字")]]
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
    # log_callback()
