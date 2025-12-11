# cache：https://reference.langchain.com/python/langchain_core/caches/?h=cache#langchain_core.caches
# LangChain的缓存
# 1.缓存是LangChain中一个重要的概念，它允许您在执行昂贵的LLM调用时存储和重用结果。这可以显著提高性能，尤其是在需要多次调用相同输入的情况下。
# 2.缓存可以减少LLM的调用，从而加快应用响应速度
#
# https://lagnchain.readthedocs.io/en/stable/modules/memory/types/buffer_window.html
# 缓存类型(应该是已被废弃)
# 1.ConversationBufferMemory:
# This memory allows for storing of messages and then extracts the messages in a variable.
#
# 2.ConversationBufferWindowMemory:
# keeps a list of the interactions of the conversation over time. It only uses the last K interactions.
# This can be useful for keeping a sliding window of the most recent interactions, so the buffer does not get too large
#
# 3.ConversationSummaryMemory:
# This type of memory creates a summary of the conversation over time. This can be useful for condensing information from the conversation over time.
#
# 4.ConversationSummaryBufferMemory:It keeps a buffer of recent interactions in memory,
# but rather than just completely flushing old interactions it compiles them into a summary and uses both. Unlike the previous implementation though,
# it uses token length rather than number of interactions to determine when to flush interactions.
#
# 5.ConversationTokenBufferMemory:keeps a buffer of recent interactions in memory, and uses token length rather than number of interactions to determine when to flush interactions.
#
# 6.VectorStore-Backed Memory:stores memories in a VectorDB and queries the top-K most “salient” docs every time it is called.
# This differs from most of the other Memory classes in that it doesn’t explicitly track the order of interactions.
# In this case, the “docs” are previous conversation snippets. This can be useful to refer to relevant pieces of information that the AI was told earlier in the conversation.

import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

OPEN_API_URL = os.environ.get("OPEN_API_URL")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")

# 1.缓冲记忆
#
if OPEN_API_KEY is None:
    raise ValueError("OPENAI_KEY_V4 is not set")
llm = ChatOpenAI(
    base_url=OPEN_API_URL,
    api_key=SecretStr(OPEN_API_KEY),
)

# 初始化聊天历史
chat_history = InMemoryChatMessageHistory()
# 创建提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

# 创建链
chain = prompt | llm
# 添加消息历史功能
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

response = chain_with_history.invoke(
    {"input": "你好，我叫张三"}, config={"configurable": {"session_id": "user123"}}
)
print(response.content)

# 继续对话
response = chain_with_history.invoke(
    {"input": "我刚才说我叫什么"}, config={"configurable": {"session_id": "user123"}}
)
print(response.content)
