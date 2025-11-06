#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, Iterable, List, Optional, Iterator, Mapping
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage, HumanMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration,ChatGenerationChunk,ChatResult
from pydantic import Field

# 自定义模型：langchain允许通过继承其基类来轻松创建自己的模型封装
# 为什么要创建自定义模型
# 1.集成新模型：为新模型提供一个标准的LangChain接口
# 2.修改现有行为：封装一个现有的模型，并在其之上添加自定义逻辑，例如特定的重试策略、请求日志、默认参数修改等
# 3.用于测试：创建一个可预测的、假的模型，用于单元测试和集成测试你的链和代理，而无需实际调用昂贵的LLM API

class ChatParrotLink(BaseChatModel):
    """
    A custom chat model that echoes the first `parrot_buffer_length` characters
    of tht input

    """
    model_name: str = Field(alias="model")
    parrot_buffer_length: int
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        重写_generate方法，实现自定义chat逻辑
        可实现请求api，可调用本地模型，或者其他逻辑
        :param messages:
        :param stop:
        :param run_manager:
        :param kwargs:
        :return:
        """
        last_message = messages[-1]
        tokens = last_message.content[:self.parrot_buffer_length]
        ct_input_tokens = sum(len(message.content) for message in messages)
        ct_output_tokens = len(tokens)
        message = AIMessage(
            content=tokens,
            additional_kwargs={},# Used to add additional payload to the message
            response_metadata={
                "time_in_seconds":3,
                "model_name":self.model_name,
            },
            usage_metadata={
                "input_tokens":ct_input_tokens,
                "output_tokens":ct_output_tokens,
                "total_tokens":ct_input_tokens + ct_output_tokens,
            }
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream the output of the model.
        This method should be implemented if the model can generate output
        in a streaming fashion.If the model does not support streaming,do not
        implement it,In that case streaming requests will be automatically handled
        by the _generate method.

        :param messages:
        :param stop:
        :param run_manager:
        :param kwargs:
        :return:
        """
        last_message = messages[-1]
        tokens = str(last_message.content[:self.parrot_buffer_length])
        ct_input_tokens = sum(len(message.content) for message in messages)
        for token in tokens:
            usage_metadata = UsageMetadata(
                {
                    "input_tokens":ct_input_tokens,
                    "output_tokens":1,
                    "total_tokens":ct_input_tokens + 1,
                }
            )
            ct_input_tokens = 0
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=token,
                    usage_metadata=usage_metadata,
                )
            )
            if run_manager:
                run_manager.on_llm_new_token(token,chunk=chunk)

            yield chunk

        # ad some other information
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                response_metadata={
                    "time_in_sec":3,
                    "model_name":self.model_name,
                }
            )
        )
        if run_manager:
            run_manager.on_llm_new_token(token,chunk=chunk)
        yield chunk

    @property
    def _llm_type(self) -> str:
        """
        Get the type of language model used by this chat model.
        :return:
        """
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Return a dictionary of identifying parameters.
        This information is used by LangChain callback system,which
        is used for tracing purposes make if possible to monitor LLMs.
        :return:
        """
        return {
            "model_name":self.model_name,
        }


def main():
    model = ChatParrotLink(
        parrot_buffer_length=3,
        model="my_custom_model",
    )
    response = model.invoke(
        [
            HumanMessage(content="hello"),
            AIMessage(content="Hi there human!"),
            HumanMessage(content="Meow!"),
        ]
    )
    print(f"----1:{response}")
    response = model.invoke("hello")
    print(f"----2: {response}")

    response = model.batch(['hello','goodbye'])
    print(f"----3: {response}")

    for chunk in model.stream('cat'):
        print(chunk.content,end='|')

if __name__ == "__main__":
    main()



