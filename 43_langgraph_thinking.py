# 使用LangGraph构建智能体时，你首先需要将其拆解为一个独立的孤舟，这些步骤被称为节点
# 接着需要定义每个节点对应的各类决策逻辑与状态转换规则。最后，通过一个共享状态将所有节点连接起来
# 每个节点都能对这个共享状态进行读写操作

# 假设需要构建一个处理客户支持邮件的智能体，团队提供了一些需求
# The agent should:

# - Read incoming customer emails
# - Classify them by urgency and topic
# - Search relevant documentation to answer questions
# - Draft appropriate responses
# - Escalate complex issues to human agents
# - Schedule follow-ups when needed

# Example scenarios to handle:

# 1. Simple product question: "How do I reset my password?"
# 2. Bug report: "The export feature crashes when I select PDF format"
# 3. Urgent billing issue: "I was charged twice for my subscription!"
# 4. Feature request: "Can you add dark mode to the mobile app?"
# 5. Complex technical issue: "Our API integration fails intermittently with 504 errors"

# 使用LangGraph来实现上述智能体，需要一下五个步骤
# 1.将工作流程划分为独立步骤
# 首先，明确流程中的各个不同步骤，每个步骤都将称为一个节点。接下来描绘这些步骤之间的相互关系
#
# Now that we’ve identified the components in our workflow, let’s understand what each node needs to do:

# Read Email: Extract and parse the email content
# Classify Intent: Use an LLM to categorize urgency and topic, then route to appropriate action
# Doc Search: Query your knowledge base for relevant information
# Bug Track: Create or update issue in tracking system
# Draft Reply: Generate an appropriate response
# Human Review: Escalate to human agent for approval or handling
# Send Reply: Dispatch the email response
#
#
# 2.定义每个步骤所要处理的事
#

# 3.设计状态
#
# 4.构建节点
#
# 5.串联节点
