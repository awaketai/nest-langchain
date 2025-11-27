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
