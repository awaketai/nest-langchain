# 大语言模型本身是无状态的，这意味这每次向LLM发送一个请求时，它都会忘记之前的所有交互
# 这对于构建聊天机器人、代理或者需要上下文感知的应用来说，这种无状态是致命的
# 记忆(memory)是LangChain解决这个问题的核心组件。它允许LLM应用程序记住过去的交互，从而
# 在多轮对话中保持长下文和连贯性
#
# 为什么记忆重要
# 1.上下文感知：聊天机器人需要记住用户之前说过什么，才能理解后续的问题并给出相关的回答
# 2.个性化：代理可以记住用户的偏好或历史行为，从而提供更个性化的服务
# 3.避免重复：代理可以记住它已经执行过的操作或者已经获取过的信息，避免重复劳动
# 4.学习：代理可以从过去的成功或失败中学习，从而改进其未来的决策
#
# 记忆的工作原理
# LangChain的就组件通常通过管理一个chat_history变量来实现。这个变量存储了HumangMessage和
# AIMessage的列表。当链或者代理被调用时，记忆组件会将这个chat_history注入到提示中，从而为
# LLM提供上下文
#
# 常见的记忆类型
# 1.ChatMessageHistory(基础消息历史)：
# 1.1 特点：最基础的聊天消息存储方式，直接存储HumanMessage和AIMessage对象的列表，它是构建所有更复杂记忆类型的基础
# 1.2 有点：灵活、直接、易于理解和操作
# 1.3 缺点：自身不提供高级管理策略(如限制长度、总结等)，需要手动处理
#
# 2.ConversationBufferMemory(缓冲区记忆)旧版/推荐使用ChatMessageHistory构建
# 2.1 特点：最简单的记忆类型。它能将所有对话历史(原始消息)完整的存储在一个缓冲区中
# 2.2 优点：简单易用，保留所有细节
# 2.3 缺点：随着对话轮数增加，历史会变的非常长，可能超出LLM的上下文窗口限制，并增加成本
#
# 3.ConversationBufferWindowMemory(滑动窗口记忆)旧版/推荐使用ChatMessageHistory构建
# 3.1 特点：存储最近K轮对话。当对话超过K轮时，最旧的对话会被丢弃
# 3.2 优点：有效控制上下文长度，避免超出LLM限制
# 3.3 缺点：丢失旧的上下文
#
# 4.ConversationSummaryMemory(摘要记忆)
# 4.1 特点：使用一个LLM来总结过去的对话，它不会存储原始消息，而是存储一个不断更新的对话摘要
# 4.2 优点：适用与非常长的对话，可以大大减少上下文长度
# 4.3 缺点：总结过程本身会消耗LLM资源，并且可能会丢失一些细节
#
# 5.ConversationSummaryBufferMemory(摘要缓冲区记忆)
# 5.1 特点：结合了ConversationBufferWindowMemory和ConversationSummaryMemory。它会存储最近K轮的原始对话，而更早的对话会被总结
# 5.2 优点：兼顾了细节和上下文长度控制
#
# 6.ConversationKGMemory(知识图谱记忆)
# 6.1 特点：使用LLM从对话中提取实体和关系，构建一个知识图谱。当需要上下文时，它会查询知识图谱
# 6.2 优点：能够记住对话中的事实，即使对话很长
# 6.3 缺点：复杂性较高，需要额外的LLM调用来构建和查询知识图谱
#
# 7.VectorStoreRetrieverMemory(向量存储检索记忆)
# 7.1 特点：将对话历史的每个回合都嵌入并存储在向量存储中，当需要上下文时，它会根据当前输入检索最相关的历史回合
# 7.2 优点：适用于非常长的对话，可以检索到最相关的历史片段
# 7.3 缺点：需要嵌入模型和向量存储
#
# 示例1：自定义缓冲区记忆
# 基于ChatMessageHistory构建一个自定义的缓冲区记忆SimpleBufferMemory，模拟传统ConversationBufferMemory的核心功能，即存储完整的对话功能
# 核心概念：
# SimpleBufferMemory类：一个自定义类，内部使用ChatMessageHistory来存储每个会话的聊天记录
# get_session_history:根据session_id获取或创建ChatMessageHistory实例
# add_conversation:模拟ConversationBufferMemory的save_context方法，将用户输入和AI响应添加到历史中
# get_messages:获取指定会话的所有消息
# get_conversation_string:将消息列表格式化为可读的字符串，类似与旧版ConversationBufferMemory的buffer属性
#
# 示例亮点
# 模块化流程：展示了如何利用ChatMessageHistory作为基础构建更复杂的记忆逻辑
# 清晰的流程：演示了如何添加对话、检索历史以及将历史格式化输出
# 适应新架构：提供了在不直接依赖旧版ConversationBUfferMemory的情况相爱，实现相同功能的思路
#
# 示例2：基础消息历史
# 展示了ChatMessageHistory的基本用法，它是LangChain中管理对话消息最直接的方式
#
# 核心概念：
# ChatMessageHistory实例化：直接创建一个ChatMessageHistory对象来管理一个会话的消息
# add_user_message:添加用户的消息到历史中
# add_ai_message:添加AI的响应到历史中
# history.messages:直接访问存储在历史中的消息列表，其中包含HUmanMessage和AIMessage对象
#
# 实例亮点：
# 简洁性：演示了ChatMessageHistory作为独立组件的简洁用法
# 消息类型：强调了HUmanMessage和AIMessage在构建对话历史中的作用
# 基础构建块：说明了ChatMessageHistory是所有更高级记忆类型的基础
#


import os
from typing import Any, Dict, List

import dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

dotenv.load_dotenv()

OPEN_API_URL = os.environ.get("OPEN_API_URL")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")


class SimpleBufferMemory:
    """
    简单的缓冲区记忆类，基于ChatMessageHistory实现
    模拟ConversationBufferMemory的核心功能
    """

    def __init__(
        self,
        chat_memory: BaseChatMessageHistory = None,
        return_messages: bool = False,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
    ) -> None:
        """
        初始化SimpleBufferMemory
        Args:
            chat_memory: 聊天消息历史对象，默认使用 ChatMessageHistory
            return_messages: 是否返回消息对象列表，False 则返回字符串
            memory_key: 存储历史的键
            input_key: 存储用户输入的键
            output_key: 存储AI输出的键
            human_prefix: 用户消息的前缀
            ai_prefix: AI消息的前缀
        """
        self.chat_memory = chat_memory or ChatMessageHistory()
        self.return_messages = return_messages
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        保存对话上下文到记忆中
        Args:
            inputs: 输入字典，包含用户输入
            outputs: 输出字典，包含 AI 响应
        """
        # 从输入中提取用户消息
        input_str = inputs.get(self.input_key, "")
        # 从输出中提取AI相应
        output_str = outputs.get(self.output_key, "")
        # 添加到聊天历史
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        加载记忆变量
        Args:
            inputs: 输入字典(可选)
        Returns:
            Dict[str,Any]: 记忆变量字典
        """
        if self.return_messages:
            # 返回消息对象列表
            return {self.memory_key: self.chat_memory.messages}
        else:
            # 返回格式化的字符串
            return {self.memory_key: self._get_buffer_string()}

    def _get_buffer_string(self) -> str:
        """
        将消息历史转化为格式化字符串
        """
        messages = self.chat_memory.messages
        string_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                prefix = self.human_prefix
            elif isinstance(msg, AIMessage):
                prefix = self.ai_prefix
            else:
                prefix = msg.__class__.__name__

            string_messages.append(f"{prefix}: {msg.content}")

        return "\n".join(string_messages)

    def clear(self):
        """
        清空记忆
        """
        self.chat_memory.clear()

    @property
    def messages(self) -> List[BaseMessage]:
        """
        获取所有消息
        """
        return self.chat_memory.messages

    def format_messages(self) -> str:
        """
        将消息格式化为字符串

        Returns:
            格式化的对话历史字符串
        """
        messages = self.messages
        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"AI: {msg.content}")
        return "\n".join(formatted)


def simple_buffer_memory():
    print("=== SimpleBufferMemory 示例 ===\n")

    # 1. 基本使用 - 返回字符串格式
    print("1. 基本使用（字符串格式）:")
    memory = SimpleBufferMemory()

    # 保存对话上下文
    memory.save_context(
        {"input": "你好，我叫小明"}, {"output": "你好小明！很高兴认识你。"}
    )

    memory.save_context(
        {"input": "你能帮我写代码吗？"},
        {"output": "当然可以！我很乐意帮你写代码。你想写什么样的代码？"},
    )

    memory.save_context(
        {"input": "帮我写一个 Python 的 for 循环"},
        {
            "output": "好的，这是一个简单的 for 循环示例：\nfor i in range(10):\n    print(i)"
        },
    )

    # 加载记忆
    variables = memory.load_memory_variables({})
    print(variables[memory.memory_key])
    print()

    # 2. 返回消息对象列表
    print("\n2. 返回消息对象列表:")
    memory_with_messages = SimpleBufferMemory(return_messages=True)

    memory_with_messages.save_context(
        {"input": "什么是机器学习？"},
        {"output": "机器学习是人工智能的一个分支，它让计算机能够从数据中学习。"},
    )

    variables = memory_with_messages.load_memory_variables({})
    messages = variables[memory_with_messages.memory_key]

    for msg in messages:
        print(f"{msg.__class__.__name__}: {msg.content}")
    print()

    # 3. 自定义前缀
    print("\n3. 自定义前缀:")
    custom_memory = SimpleBufferMemory(human_prefix="用户", ai_prefix="助手")

    custom_memory.save_context(
        {"input": "今天天气怎么样？"}, {"output": "抱歉，我无法获取实时天气信息。"}
    )

    variables = custom_memory.load_memory_variables({})
    print(variables[custom_memory.memory_key])
    print()

    # 4. 清空记忆
    print("\n4. 清空记忆:")
    print(f"清空前消息数量: {len(memory.messages)}")
    memory.clear()
    print(f"清空后消息数量: {len(memory.messages)}")


def create_conversation_chain(llm, memory: SimpleBufferMemory):
    """
    创建带记忆的对话链
    Args:
        llm:大语言模型
        memory: SimpleBufferMemory
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个友好的AI助手，请根据对话历史回答用户的问题"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    # 创建
    chain = prompt | llm

    return chain


def run_conversation(llm, memory: SimpleBufferMemory, user_input: str) -> str:
    chain = create_conversation_chain(llm, memory)
    # 加载历史消息
    memory_vars = memory.load_memory_variables({})
    # 准备输入
    chain_input = {"history": memory_vars["history"], "input": user_input}
    # 调用模型
    response = chain.invoke(chain_input)
    # 保存上下文
    memory.save_context({"input": user_input}, {"output": response.content})

    # 返回模型输出
    return response.content


def main():
    if OPEN_API_KEY is None:
        raise ValueError("OPENAI_KEY_V4 is not set")
    llm = ChatOpenAI(
        base_url=OPEN_API_URL,
        api_key=SecretStr(OPEN_API_KEY),
        model="gpt-4",
    )
    # 创建新的记忆实例
    chat_history_2 = ChatMessageHistory()
    memory_2 = SimpleBufferMemory(chat_memory=chat_history_2, return_messages=True)

    # 进行对话
    response_1 = run_conversation(llm, memory_2, "你好，我叫张三")
    print(f"用户: 你好，我叫张三")
    print(f"AI: {response_1}\n")

    response_2 = run_conversation(llm, memory_2, "我喜欢打篮球")
    print(f"用户: 我喜欢打篮球")
    print(f"AI: {response_2}\n")

    response_3 = run_conversation(llm, memory_2, "你还记得我的名字和爱好吗？")
    print(f"用户: 你还记得我的名字和爱好吗？")
    print(f"AI: {response_3}\n")

    # 显示历史
    print("=" * 60)
    print("对话历史：")
    print("=" * 60)
    print(memory_2.format_messages)


if __name__ == "__main__":
    main()
