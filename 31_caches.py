# cache：https://reference.langchain.com/python/langchain_core/caches/?h=cache#langchain_core.caches
# LangChain的缓存
# 1.缓存是LangChain中一个重要的概念，它允许您在执行昂贵的LLM调用时存储和重用结果。这可以显著提高性能，尤其是在需要多次调用相同输入的情况下。
# 2.缓存可以减少LLM的调用，从而加快应用响应速度
#
# https://lagnchain.readthedocs.io/en/stable/modules/memory/types/buffer_window.html
# 缓存类型
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
