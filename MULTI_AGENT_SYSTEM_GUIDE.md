# 多智能体协调系统完整开发指南

## 目录
1. [核心概念与架构](#1-核心概念与架构)
2. [消息协议设计](#2-消息协议设计)
3. [智能体实现](#3-智能体实现)
4. [通信模式](#4-通信模式)
5. [设计模式](#5-设计模式)
6. [运行时环境](#6-运行时环境)
7. [完整实现示例](#7-完整实现示例)
8. [最佳实践](#8-最佳实践)
9. [常见问题与解决方案](#9-常见问题与解决方案)

---

## 1. 核心概念与架构

### 1.1 什么是多智能体系统

多智能体系统（Multi-Agent System）是由多个自主的智能体（Agent）组成的系统，这些智能体通过消息传递进行交互，共同完成复杂任务。

**智能体的核心特征：**
- **消息驱动**: 通过接收和发送消息进行通信
- **状态管理**: 维护自己的内部状态
- **自主行动**: 根据消息和状态执行特定操作
- **模块化**: 可以独立开发、测试和部署

**多智能体系统的优势：**
- 任务分解：将复杂问题分解为多个子任务
- 专业化：每个智能体专注于特定领域
- 可扩展性：易于添加新的智能体
- 容错性：单个智能体失败不影响整体系统
- 分布式部署：可以在不同机器上运行

### 1.2 AutoGen架构概览

AutoGen提供了三层架构：

```
┌─────────────────────────────────────────────────────────┐
│           Application Layer (应用层)                     │
│  - 业务逻辑                                              │
│  - 工作流编排                                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│          Component Layer (组件层)                        │
│  - Routed Agents (路由智能体)                            │
│  - Model Clients (模型客户端)                            │
│  - Tools (工具)                                          │
│  - Code Executors (代码执行器)                           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│          Runtime Layer (运行时层)                        │
│  - SingleThreadedAgentRuntime (单线程运行时)             │
│  - DistributedAgentRuntime (分布式运行时)                │
│  - Message Routing (消息路由)                            │
│  - Lifecycle Management (生命周期管理)                   │
└─────────────────────────────────────────────────────────┘
```

### 1.3 Agent ID 与 Agent Type

**理解 Agent ID (智能体标识符):**

Agent ID 由两部分组成：
- **Agent Type (类型)**: 标识智能体的类别，如 "CoderAgent", "ReviewerAgent"
- **Agent Key (键)**: 标识同一类型智能体的不同实例，如 "session_123"

```python
from autogen_core import AgentId

# 创建一个Agent ID
agent_id = AgentId(type="CoderAgent", key="session_001")
```

**为什么需要这种设计？**
- 支持多租户场景（multi-tenant）
- 每个用户会话可以有独立的智能体实例
- 便于状态隔离和并发处理

### 1.4 Topic 与 Subscription

**Topic (主题)** 是广播消息的作用域：

```python
from autogen_core import TopicId

# Topic 也由两部分组成
topic_id = TopicId(type="CodeReview", source="session_001")
```

**Subscription (订阅)** 将主题映射到智能体：

```python
from autogen_core import TypeSubscription

# Type-based Subscription: 将主题类型映射到智能体类型
subscription = TypeSubscription(
    topic_type="CodeReview",
    agent_type="ReviewerAgent"
)
```

**订阅模式的优势：**
- 松耦合：发送者不需要知道接收者的具体ID
- 灵活性：可以动态添加或移除订阅
- 可扩展性：支持一对多的消息传递

---

## 2. 消息协议设计

### 2.1 消息类型定义

消息是智能体之间通信的基础。使用Python的`dataclass`定义消息类型：

```python
from dataclasses import dataclass
from typing import List
from autogen_core.models import LLMMessage

@dataclass
class TaskRequest:
    """任务请求消息"""
    task_id: str
    description: str
    priority: int
    requester: str

@dataclass
class TaskResponse:
    """任务响应消息"""
    task_id: str
    result: str
    status: str  # "completed", "failed", "in_progress"
    agent_id: str

@dataclass
class TaskDelegation:
    """任务委托消息"""
    task_id: str
    context: List[LLMMessage]
    target_agent_type: str
    metadata: dict
```

### 2.2 消息协议设计原则

**1. 明确的消息流向**

设计消息协议前，先画出数据流图：

```
User → TaskRequest → CoordinatorAgent
                          ↓
            ┌─────────────┴─────────────┐
            ↓                           ↓
    SubTaskRequest              SubTaskRequest
            ↓                           ↓
    WorkerAgentA                WorkerAgentB
            ↓                           ↓
      SubTaskResult             SubTaskResult
            └─────────────┬─────────────┘
                          ↓
                  CoordinatorAgent
                          ↓
                   TaskResponse
                          ↓
                        User
```

**2. 包含必要的上下文信息**

消息应该包含足够的信息让接收者理解和处理：

```python
@dataclass
class CodeReviewRequest:
    session_id: str              # 会话标识
    code: str                    # 待审查的代码
    language: str                # 编程语言
    context: List[LLMMessage]    # 历史对话上下文
    requirements: List[str]      # 审查要求
    previous_feedback: str       # 之前的反馈（如果有）
```

**3. 支持追溯和调试**

```python
@dataclass
class Message:
    message_id: str              # 消息唯一ID
    timestamp: float             # 时间戳
    sender_id: str               # 发送者ID
    correlation_id: str          # 关联ID（用于追踪整个请求链）
    payload: dict                # 实际数据
```

### 2.3 消息协议示例：客户服务系统

```python
from dataclasses import dataclass
from typing import List, Literal
from autogen_core.models import LLMMessage

@dataclass
class CustomerQuery:
    """客户查询"""
    query_id: str
    customer_id: str
    message: str
    channel: Literal["web", "mobile", "phone"]

@dataclass
class TriageDecision:
    """分流决策"""
    query_id: str
    category: Literal["sales", "support", "refund", "human"]
    confidence: float
    reasoning: str

@dataclass
class AgentHandoff:
    """智能体交接"""
    query_id: str
    from_agent: str
    to_agent: str
    conversation_history: List[LLMMessage]
    handoff_reason: str

@dataclass
class ResolutionComplete:
    """问题解决"""
    query_id: str
    resolution: str
    customer_satisfied: bool
    handling_agent: str
```

---

## 3. 智能体实现

### 3.1 基础智能体结构

所有智能体都应该继承`RoutedAgent`并使用`@message_handler`装饰器处理消息：

```python
from autogen_core import RoutedAgent, MessageContext, message_handler
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage

class BaseAIAgent(RoutedAgent):
    """基础AI智能体"""

    def __init__(
        self,
        description: str,
        model_client: ChatCompletionClient,
        system_prompt: str
    ):
        super().__init__(description)
        self._model_client = model_client
        self._system_message = SystemMessage(content=system_prompt)

    async def _generate_response(
        self,
        user_message: str,
        ctx: MessageContext
    ) -> str:
        """调用LLM生成响应"""
        messages = [
            self._system_message,
            UserMessage(content=user_message, source="user")
        ]
        response = await self._model_client.create(
            messages=messages,
            cancellation_token=ctx.cancellation_token
        )
        return response.content
```

### 3.2 实现消息处理器

**单一消息类型处理：**

```python
@message_handler
async def handle_task_request(
    self,
    message: TaskRequest,
    ctx: MessageContext
) -> None:
    """处理任务请求"""
    print(f"[{self.id.type}] Received task: {message.task_id}")

    # 1. 处理任务
    result = await self._process_task(message, ctx)

    # 2. 发送响应
    response = TaskResponse(
        task_id=message.task_id,
        result=result,
        status="completed",
        agent_id=str(self.id)
    )

    # 3. 发布消息到topic
    await self.publish_message(
        response,
        topic_id=TopicId("TaskResponse", source=self.id.key)
    )
```

**多消息类型处理（使用Union）：**

```python
from typing import Union

@message_handler
async def handle_multiple_message_types(
    self,
    message: Union[TaskRequest, TaskUpdate, TaskCancellation],
    ctx: MessageContext
) -> None:
    """处理多种消息类型"""
    if isinstance(message, TaskRequest):
        await self._handle_new_task(message, ctx)
    elif isinstance(message, TaskUpdate):
        await self._handle_update(message, ctx)
    elif isinstance(message, TaskCancellation):
        await self._handle_cancellation(message, ctx)
```

### 3.3 智能体状态管理

**会话级状态管理：**

```python
from typing import Dict, List

class StatefulAgent(RoutedAgent):
    """带状态管理的智能体"""

    def __init__(self, description: str):
        super().__init__(description)
        # 使用字典存储不同会话的状态
        self._session_states: Dict[str, SessionState] = {}

    @message_handler
    async def handle_message(
        self,
        message: ConversationMessage,
        ctx: MessageContext
    ) -> None:
        # 获取或创建会话状态
        if message.session_id not in self._session_states:
            self._session_states[message.session_id] = SessionState()

        state = self._session_states[message.session_id]

        # 更新状态
        state.message_count += 1
        state.history.append(message)

        # 处理消息
        response = await self._process_with_state(message, state, ctx)

        # 发送响应
        await self.publish_message(response, topic_id=...)

@dataclass
class SessionState:
    """会话状态"""
    message_count: int = 0
    history: List[ConversationMessage] = None
    context_data: dict = None

    def __post_init__(self):
        if self.history is None:
            self.history = []
        if self.context_data is None:
            self.context_data = {}
```

### 3.4 工具使用（Tool Use）

智能体可以使用工具来完成特定任务：

```python
from autogen_core.tools import FunctionTool

# 定义工具函数
def search_database(query: str, limit: int = 10) -> List[dict]:
    """搜索数据库"""
    # 实际的数据库搜索逻辑
    return [{"id": 1, "content": "..."}]

def send_email(recipient: str, subject: str, body: str) -> bool:
    """发送邮件"""
    # 实际的邮件发送逻辑
    return True

# 创建工具
search_tool = FunctionTool(
    search_database,
    description="Search the database for relevant information"
)

email_tool = FunctionTool(
    send_email,
    description="Send an email to a recipient"
)

# 在智能体中使用工具
class ToolUsingAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient):
        super().__init__("Tool using agent")
        self._model_client = model_client
        self._tools = {
            "search_database": search_tool,
            "send_email": email_tool
        }
        self._tool_schemas = [tool.schema for tool in self._tools.values()]

    @message_handler
    async def handle_request(
        self,
        message: Request,
        ctx: MessageContext
    ) -> None:
        # 调用LLM并提供工具
        response = await self._model_client.create(
            messages=[UserMessage(content=message.text, source="user")],
            tools=self._tool_schemas,
            cancellation_token=ctx.cancellation_token
        )

        # 处理工具调用
        if isinstance(response.content, list):
            for call in response.content:
                if isinstance(call, FunctionCall):
                    tool = self._tools[call.name]
                    result = await tool.run_json(
                        json.loads(call.arguments),
                        ctx.cancellation_token
                    )
                    # 处理工具结果...
```

---

## 4. 通信模式

### 4.1 直接消息传递（Direct Messaging）

**点对点通信：**

```python
from autogen_core import AgentId

# 发送消息到特定智能体
target_agent_id = AgentId(type="WorkerAgent", key="worker_001")
await runtime.send_message(
    message=TaskRequest(task_id="123", description="Do something"),
    recipient=target_agent_id
)
```

**使用场景：**
- 明确知道接收者身份
- 一对一的请求-响应模式
- 需要精确控制消息流向

### 4.2 广播消息（Broadcast）

**发布-订阅模式：**

```python
from autogen_core import TopicId

# 发布消息到主题
await self.publish_message(
    message=EventNotification(event_type="task_completed"),
    topic_id=TopicId(type="TaskEvents", source="system")
)

# 所有订阅了"TaskEvents"主题的智能体都会收到消息
```

**使用场景：**
- 事件通知
- 一对多通信
- 发送者不需要知道接收者

### 4.3 Type-Based Subscription

**单租户，单主题场景：**

```python
# 所有智能体订阅同一个主题
await runtime.add_subscription(
    TypeSubscription(topic_type="default", agent_type="AgentA")
)
await runtime.add_subscription(
    TypeSubscription(topic_type="default", agent_type="AgentB")
)
await runtime.add_subscription(
    TypeSubscription(topic_type="default", agent_type="AgentC")
)

# 发布消息时使用相同的topic source
await runtime.publish_message(
    message=BroadcastMessage(content="Hello all"),
    topic_id=TopicId(type="default", source="default")
)
# 所有智能体实例 (AgentA:default, AgentB:default, AgentC:default) 都会收到
```

**单租户，多主题场景：**

```python
# 不同智能体订阅不同主题
await runtime.add_subscription(
    TypeSubscription(topic_type="coding_tasks", agent_type="CoderAgent")
)
await runtime.add_subscription(
    TypeSubscription(topic_type="review_tasks", agent_type="ReviewerAgent")
)

# 发布到不同主题
await runtime.publish_message(
    message=CodingTask(...),
    topic_id=TopicId(type="coding_tasks", source="default")
)
# 只有CoderAgent:default会收到

await runtime.publish_message(
    message=ReviewTask(...),
    topic_id=TopicId(type="review_tasks", source="default")
)
# 只有ReviewerAgent:default会收到
```

**多租户场景：**

```python
# 一个type-based subscription
await runtime.add_subscription(
    TypeSubscription(topic_type="user_queries", agent_type="AssistantAgent")
)

# 不同用户会话使用不同的source
# 用户1
await runtime.publish_message(
    message=Query(text="Hello"),
    topic_id=TopicId(type="user_queries", source="user_session_001")
)
# 创建 AssistantAgent:user_session_001

# 用户2
await runtime.publish_message(
    message=Query(text="Hi"),
    topic_id=TopicId(type="user_queries", source="user_session_002")
)
# 创建 AssistantAgent:user_session_002

# 两个用户有独立的智能体实例，状态完全隔离
```

---

## 5. 设计模式

### 5.1 Handoff Pattern (交接模式)

**核心思想：** 智能体通过特殊的工具调用将任务委托给其他智能体

**实现步骤：**

**步骤1: 定义委托工具**

```python
def transfer_to_sales() -> str:
    """Transfer to sales team"""
    return "SalesAgent"  # 返回目标智能体的topic type

def transfer_to_support() -> str:
    """Transfer to support team"""
    return "SupportAgent"

transfer_to_sales_tool = FunctionTool(
    transfer_to_sales,
    description="Use when customer wants to buy something"
)

transfer_to_support_tool = FunctionTool(
    transfer_to_support,
    description="Use when customer needs technical support"
)
```

**步骤2: 创建可以委托任务的智能体**

```python
class HandoffAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        model_client: ChatCompletionClient,
        regular_tools: List[Tool],
        handoff_tools: List[Tool],
        agent_topic_type: str,
        user_topic_type: str
    ):
        super().__init__(description)
        self._model_client = model_client
        self._regular_tools = {t.name: t for t in regular_tools}
        self._handoff_tools = {t.name: t for t in handoff_tools}
        self._agent_topic_type = agent_topic_type
        self._user_topic_type = user_topic_type

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        # 调用LLM
        response = await self._model_client.create(
            messages=[SystemMessage(...)] + message.context,
            tools=[t.schema for t in self._regular_tools.values()] +
                  [t.schema for t in self._handoff_tools.values()],
            cancellation_token=ctx.cancellation_token
        )

        # 处理响应
        if isinstance(response.content, list):  # 工具调用
            for call in response.content:
                if call.name in self._handoff_tools:
                    # 委托给另一个智能体
                    target_topic = await self._handoff_tools[call.name].run_json(
                        json.loads(call.arguments),
                        ctx.cancellation_token
                    )

                    # 发布到目标智能体的主题
                    await self.publish_message(
                        UserTask(context=message.context + [response]),
                        topic_id=TopicId(type=target_topic, source=self.id.key)
                    )
                    return

                elif call.name in self._regular_tools:
                    # 执行常规工具
                    result = await self._regular_tools[call.name].run_json(...)
                    # 继续处理...

        else:  # 文本响应
            # 返回给用户
            await self.publish_message(
                AgentResponse(content=response.content),
                topic_id=TopicId(type=self._user_topic_type, source=self.id.key)
            )
```

**步骤3: 设置订阅**

```python
# 每个智能体订阅自己的主题
await runtime.add_subscription(
    TypeSubscription(topic_type="TriageAgent", agent_type="TriageAgent")
)
await runtime.add_subscription(
    TypeSubscription(topic_type="SalesAgent", agent_type="SalesAgent")
)
await runtime.add_subscription(
    TypeSubscription(topic_type="SupportAgent", agent_type="SupportAgent")
)
```

### 5.2 Reflection Pattern (反思模式)

**核心思想：** 一个智能体生成内容，另一个智能体进行评审和反馈，循环直到满意

**完整实现：**

```python
from dataclasses import dataclass
from typing import List

@dataclass
class GenerationTask:
    """生成任务"""
    task_id: str
    requirement: str

@dataclass
class ReviewRequest:
    """评审请求"""
    task_id: str
    content: str
    iteration: int
    history: List[str]

@dataclass
class ReviewResult:
    """评审结果"""
    task_id: str
    feedback: str
    approved: bool
    iteration: int

@dataclass
class FinalResult:
    """最终结果"""
    task_id: str
    content: str
    iterations: int

class GeneratorAgent(RoutedAgent):
    """内容生成器"""

    def __init__(self, model_client: ChatCompletionClient):
        super().__init__("Content generator")
        self._model_client = model_client
        self._sessions = {}  # 存储会话状态

    @message_handler
    async def handle_generation_task(
        self,
        message: GenerationTask,
        ctx: MessageContext
    ) -> None:
        """处理新的生成任务"""
        # 生成初始内容
        response = await self._model_client.create(
            messages=[
                SystemMessage(content="You are a content generator."),
                UserMessage(content=message.requirement, source="user")
            ],
            cancellation_token=ctx.cancellation_token
        )

        # 保存会话状态
        self._sessions[message.task_id] = {
            "requirement": message.requirement,
            "history": [],
            "iteration": 0
        }

        # 发送评审请求
        await self.publish_message(
            ReviewRequest(
                task_id=message.task_id,
                content=response.content,
                iteration=0,
                history=[]
            ),
            topic_id=TopicId("ReviewRequest", source=self.id.key)
        )

    @message_handler
    async def handle_review_result(
        self,
        message: ReviewResult,
        ctx: MessageContext
    ) -> None:
        """处理评审结果"""
        session = self._sessions[message.task_id]

        if message.approved:
            # 评审通过，发布最终结果
            await self.publish_message(
                FinalResult(
                    task_id=message.task_id,
                    content=session["history"][-1],
                    iterations=message.iteration
                ),
                topic_id=TopicId("FinalResult", source=self.id.key)
            )
            # 清理会话
            del self._sessions[message.task_id]

        else:
            # 根据反馈改进
            session["history"].append(message.feedback)

            # 构建改进提示
            improvement_messages = [
                SystemMessage(content="You are a content generator."),
                UserMessage(content=session["requirement"], source="user")
            ]

            # 添加历史反馈
            for i, feedback in enumerate(session["history"]):
                improvement_messages.append(
                    UserMessage(content=f"Feedback {i+1}: {feedback}", source="reviewer")
                )

            # 生成改进版本
            response = await self._model_client.create(
                messages=improvement_messages,
                cancellation_token=ctx.cancellation_token
            )

            # 发送新的评审请求
            await self.publish_message(
                ReviewRequest(
                    task_id=message.task_id,
                    content=response.content,
                    iteration=message.iteration + 1,
                    history=session["history"]
                ),
                topic_id=TopicId("ReviewRequest", source=self.id.key)
            )

class ReviewerAgent(RoutedAgent):
    """评审者"""

    def __init__(self, model_client: ChatCompletionClient, max_iterations: int = 3):
        super().__init__("Content reviewer")
        self._model_client = model_client
        self._max_iterations = max_iterations

    @message_handler
    async def handle_review_request(
        self,
        message: ReviewRequest,
        ctx: MessageContext
    ) -> None:
        """评审内容"""
        # 构建评审提示
        review_prompt = f"""
Review the following content:

{message.content}

Previous feedback history:
{chr(10).join(message.history) if message.history else "None"}

Provide detailed feedback on:
1. Quality
2. Completeness
3. Correctness

Format your response as JSON:
{{
    "quality_score": <1-10>,
    "feedback": "<your feedback>",
    "approved": <true/false>
}}
"""

        # 生成评审
        response = await self._model_client.create(
            messages=[
                SystemMessage(content="You are a critical reviewer."),
                UserMessage(content=review_prompt, source="system")
            ],
            json_output=True,
            cancellation_token=ctx.cancellation_token
        )

        import json
        review_data = json.loads(response.content)

        # 如果达到最大迭代次数，强制通过
        approved = review_data["approved"] or message.iteration >= self._max_iterations

        # 发送评审结果
        await self.publish_message(
            ReviewResult(
                task_id=message.task_id,
                feedback=review_data["feedback"],
                approved=approved,
                iteration=message.iteration
            ),
            topic_id=TopicId("ReviewResult", source=self.id.key)
        )
```

### 5.3 Group Chat Pattern (群聊模式)

**核心思想：** 多个智能体在共享的对话空间中协作，由管理器决定发言顺序

**实现步骤：**

```python
@dataclass
class GroupChatMessage:
    """群聊消息"""
    session_id: str
    sender: str
    content: str
    timestamp: float

@dataclass
class SpeakRequest:
    """发言请求"""
    session_id: str
    target_agent: str

class GroupChatManager(RoutedAgent):
    """群聊管理器"""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        participants: List[str]  # 参与者的agent类型列表
    ):
        super().__init__("Group chat manager")
        self._model_client = model_client
        self._participants = participants
        self._conversation_history = {}

    @message_handler
    async def handle_group_chat_message(
        self,
        message: GroupChatMessage,
        ctx: MessageContext
    ) -> None:
        """处理群聊消息"""
        # 更新对话历史
        if message.session_id not in self._conversation_history:
            self._conversation_history[message.session_id] = []

        self._conversation_history[message.session_id].append(message)

        # 决定下一个发言者
        next_speaker = await self._select_next_speaker(
            message.session_id,
            ctx
        )

        # 发送发言请求
        await self.publish_message(
            SpeakRequest(
                session_id=message.session_id,
                target_agent=next_speaker
            ),
            topic_id=TopicId(type=next_speaker, source=message.session_id)
        )

    async def _select_next_speaker(
        self,
        session_id: str,
        ctx: MessageContext
    ) -> str:
        """选择下一个发言者"""
        history = self._conversation_history[session_id]

        # 构建选择提示
        history_text = "\n".join([
            f"{msg.sender}: {msg.content}" for msg in history
        ])

        selection_prompt = f"""
Given the conversation history:
{history_text}

Available participants: {", ".join(self._participants)}

Who should speak next? Respond with just the participant name.
"""

        response = await self._model_client.create(
            messages=[
                SystemMessage(content="You coordinate group discussions."),
                UserMessage(content=selection_prompt, source="system")
            ],
            cancellation_token=ctx.cancellation_token
        )

        # 解析响应获取下一个发言者
        next_speaker = response.content.strip()

        # 验证发言者是否有效
        if next_speaker not in self._participants:
            # 默认使用轮询
            last_speaker = history[-1].sender
            current_idx = self._participants.index(last_speaker)
            next_speaker = self._participants[(current_idx + 1) % len(self._participants)]

        return next_speaker

class GroupChatParticipant(RoutedAgent):
    """群聊参与者"""

    def __init__(
        self,
        name: str,
        role: str,
        model_client: ChatCompletionClient
    ):
        super().__init__(f"Participant: {name}")
        self._name = name
        self._role = role
        self._model_client = model_client
        self._conversation_memory = {}

    @message_handler
    async def handle_group_chat_message(
        self,
        message: GroupChatMessage,
        ctx: MessageContext
    ) -> None:
        """接收群聊消息并更新记忆"""
        if message.session_id not in self._conversation_memory:
            self._conversation_memory[message.session_id] = []

        self._conversation_memory[message.session_id].append(message)

    @message_handler
    async def handle_speak_request(
        self,
        message: SpeakRequest,
        ctx: MessageContext
    ) -> None:
        """处理发言请求"""
        # 获取对话历史
        history = self._conversation_memory.get(message.session_id, [])

        # 构建上下文
        context = "\n".join([
            f"{msg.sender}: {msg.content}" for msg in history
        ])

        # 生成响应
        response = await self._model_client.create(
            messages=[
                SystemMessage(content=f"You are {self._name}, {self._role}"),
                UserMessage(content=f"Conversation:\n{context}\n\nYour turn to speak:", source="system")
            ],
            cancellation_token=ctx.cancellation_token
        )

        # 发布到群聊
        await self.publish_message(
            GroupChatMessage(
                session_id=message.session_id,
                sender=self._name,
                content=response.content,
                timestamp=time.time()
            ),
            topic_id=TopicId("GroupChat", source=message.session_id)
        )
```

---

## 6. 运行时环境

### 6.1 单线程运行时（SingleThreadedAgentRuntime）

**适用场景：**
- 本地开发和测试
- 所有智能体在同一进程中
- 简单的应用场景

**基本使用：**

```python
from autogen_core import SingleThreadedAgentRuntime

# 创建运行时
runtime = SingleThreadedAgentRuntime()

# 注册智能体类型
await MyAgent.register(
    runtime,
    type="MyAgent",
    factory=lambda: MyAgent(...)
)

# 添加订阅
await runtime.add_subscription(
    TypeSubscription(topic_type="MyTopic", agent_type="MyAgent")
)

# 启动运行时（后台处理消息）
runtime.start()

# 发送消息
await runtime.send_message(
    message=MyMessage(...),
    recipient=AgentId("MyAgent", "default")
)

# 或发布消息
await runtime.publish_message(
    message=MyMessage(...),
    topic_id=TopicId("MyTopic", "default")
)

# 等待空闲（所有消息处理完毕）
await runtime.stop_when_idle()

# 或立即停止
await runtime.stop()

# 关闭运行时释放资源
await runtime.close()
```

### 6.2 分布式运行时（DistributedAgentRuntime）

**适用场景：**
- 智能体需要分布在不同机器
- 需要横向扩展
- 跨语言的智能体系统

**架构：**

```
┌─────────────────────────────────────┐
│         Host Servicer               │
│   (消息路由、状态管理)               │
└───────┬─────────────┬───────────────┘
        │             │
┌───────▼─────┐ ┌────▼──────────┐
│  Worker 1   │ │   Worker 2    │
│  Gateway    │ │   Gateway     │
│  - AgentA   │ │   - AgentB    │
│  - AgentC   │ │   - AgentD    │
└─────────────┘ └───────────────┘
```

**Host配置：**

```python
from autogen_core.application import GrpcWorkerAgentRuntimeHost

# 创建Host
host = GrpcWorkerAgentRuntimeHost(
    address="0.0.0.0:50051"
)

# 启动Host
await host.start()

print("Host running on 0.0.0.0:50051")

# 保持运行
await host.stop_when_signal()
```

**Worker配置：**

```python
from autogen_core.application import GrpcWorkerAgentRuntime

# 创建Worker运行时
worker_runtime = GrpcWorkerAgentRuntime(
    host_address="localhost:50051"
)

# 注册智能体
await MyAgent.register(
    worker_runtime,
    type="MyAgent",
    factory=lambda: MyAgent(...)
)

# 添加订阅
await worker_runtime.add_subscription(
    TypeSubscription(topic_type="MyTopic", agent_type="MyAgent")
)

# 启动Worker
worker_runtime.start()

print("Worker connected to host")

# 保持运行
await worker_runtime.stop_when_signal()
```

### 6.3 运行时生命周期管理

**智能体生命周期：**

```python
class ManagedAgent(RoutedAgent):
    """具有完整生命周期管理的智能体"""

    async def on_activate(self) -> None:
        """智能体被激活时调用"""
        print(f"[{self.id}] Agent activated")
        # 初始化资源
        self._resources = await self._init_resources()

    async def on_deactivate(self) -> None:
        """智能体被停用时调用"""
        print(f"[{self.id}] Agent deactivated")
        # 清理资源
        await self._cleanup_resources()

    async def save_state(self) -> dict:
        """保存状态"""
        return {
            "session_data": self._session_data,
            "counter": self._counter
        }

    async def load_state(self, state: dict) -> None:
        """加载状态"""
        self._session_data = state["session_data"]
        self._counter = state["counter"]
```

---

## 7. 完整实现示例

### 7.1 示例：代码审查系统

这是一个完整的代码审查多智能体系统，包含三个智能体：
- **Coordinator**: 协调整个流程
- **Coder**: 编写代码
- **Reviewer**: 审查代码

```python
# ============================================================================
# 完整的代码审查多智能体系统
# ============================================================================

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from autogen_core import (
    AgentId,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    default_subscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ============================================================================
# 消息协议定义
# ============================================================================

@dataclass
class CodeRequest:
    """代码编写请求"""
    request_id: str
    task: str
    language: str
    requirements: List[str]

@dataclass
class CodeSubmission:
    """代码提交"""
    request_id: str
    code: str
    explanation: str
    iteration: int

@dataclass
class ReviewFeedback:
    """审查反馈"""
    request_id: str
    approved: bool
    feedback: str
    issues: List[str]
    iteration: int

@dataclass
class FinalCode:
    """最终代码"""
    request_id: str
    code: str
    iterations: int
    final_review: str

# ============================================================================
# 智能体实现
# ============================================================================

@default_subscription
class CoordinatorAgent(RoutedAgent):
    """协调器智能体 - 管理整个代码审查流程"""

    def __init__(self):
        super().__init__("Workflow coordinator")
        self._active_requests: Dict[str, dict] = {}

    @message_handler
    async def handle_code_request(
        self,
        message: CodeRequest,
        ctx: MessageContext
    ) -> None:
        """处理新的代码请求"""
        print(f"\n{'='*80}")
        print(f"[Coordinator] New code request received")
        print(f"Task: {message.task}")
        print(f"Language: {message.language}")
        print(f"Requirements: {', '.join(message.requirements)}")
        print(f"{'='*80}\n")

        # 记录请求
        self._active_requests[message.request_id] = {
            "task": message.task,
            "language": message.language,
            "requirements": message.requirements,
            "iteration": 0,
            "start_time": asyncio.get_event_loop().time()
        }

        # 转发给Coder
        await self.publish_message(
            message,
            topic_id=DefaultTopicId()
        )

    @message_handler
    async def handle_final_code(
        self,
        message: FinalCode,
        ctx: MessageContext
    ) -> None:
        """处理最终代码"""
        request_info = self._active_requests.pop(message.request_id, {})
        elapsed = asyncio.get_event_loop().time() - request_info.get("start_time", 0)

        print(f"\n{'='*80}")
        print(f"[Coordinator] Code review completed!")
        print(f"Request ID: {message.request_id}")
        print(f"Iterations: {message.iterations}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"\n--- Final Code ---")
        print(message.code)
        print(f"\n--- Final Review ---")
        print(message.final_review)
        print(f"{'='*80}\n")

@default_subscription
class CoderAgent(RoutedAgent):
    """编码智能体 - 根据任务编写代码"""

    def __init__(self, model_client: ChatCompletionClient, max_iterations: int = 3):
        super().__init__("Code writer")
        self._model_client = model_client
        self._max_iterations = max_iterations
        self._sessions: Dict[str, dict] = {}

    @message_handler
    async def handle_code_request(
        self,
        message: CodeRequest,
        ctx: MessageContext
    ) -> None:
        """处理代码编写请求"""
        print(f"[Coder] Writing code for: {message.task}")

        # 构建提示
        prompt = f"""
Write {message.language} code for the following task:
{message.task}

Requirements:
{chr(10).join(f"- {req}" for req in message.requirements)}

Provide:
1. Clean, well-documented code
2. Brief explanation of your approach

Format your response as:
EXPLANATION:
<your explanation>

CODE:
```{message.language}
<your code>
```
"""

        # 生成代码
        response = await self._model_client.create(
            messages=[
                SystemMessage(content="You are an expert programmer."),
                UserMessage(content=prompt, source="user")
            ],
            cancellation_token=ctx.cancellation_token
        )

        # 解析响应
        explanation, code = self._parse_response(response.content)

        # 保存会话状态
        self._sessions[message.request_id] = {
            "task": message.task,
            "language": message.language,
            "requirements": message.requirements,
            "history": [{"explanation": explanation, "code": code}]
        }

        print(f"[Coder] Code written (iteration 0)")

        # 提交代码审查
        await self.publish_message(
            CodeSubmission(
                request_id=message.request_id,
                code=code,
                explanation=explanation,
                iteration=0
            ),
            topic_id=DefaultTopicId()
        )

    @message_handler
    async def handle_review_feedback(
        self,
        message: ReviewFeedback,
        ctx: MessageContext
    ) -> None:
        """处理审查反馈"""
        if message.approved:
            print(f"[Coder] Code approved! ✓")
            # 已批准，不需要进一步操作
            return

        session = self._sessions[message.request_id]

        # 检查是否达到最大迭代次数
        if message.iteration >= self._max_iterations - 1:
            print(f"[Coder] Max iterations reached, submitting current version")
            # 强制提交当前版本
            last_submission = session["history"][-1]
            await self.publish_message(
                FinalCode(
                    request_id=message.request_id,
                    code=last_submission["code"],
                    iterations=message.iteration + 1,
                    final_review="Max iterations reached, force submitted"
                ),
                topic_id=DefaultTopicId()
            )
            return

        print(f"[Coder] Revising code based on feedback (iteration {message.iteration + 1})")

        # 构建改进提示
        history_text = "\n\n".join([
            f"Version {i}:\nCode:\n{h['code']}\nExplanation: {h['explanation']}"
            for i, h in enumerate(session["history"])
        ])

        revision_prompt = f"""
Original task: {session['task']}

Previous versions and feedback:
{history_text}

Latest feedback:
{message.feedback}

Issues to address:
{chr(10).join(f"- {issue}" for issue in message.issues)}

Please revise the code to address all feedback and issues.

Format your response as:
EXPLANATION:
<your explanation of changes>

CODE:
```{session['language']}
<revised code>
```
"""

        # 生成改进版本
        response = await self._model_client.create(
            messages=[
                SystemMessage(content="You are an expert programmer who learns from feedback."),
                UserMessage(content=revision_prompt, source="reviewer")
            ],
            cancellation_token=ctx.cancellation_token
        )

        # 解析响应
        explanation, code = self._parse_response(response.content)

        # 更新历史
        session["history"].append({"explanation": explanation, "code": code})

        # 提交新版本审查
        await self.publish_message(
            CodeSubmission(
                request_id=message.request_id,
                code=code,
                explanation=explanation,
                iteration=message.iteration + 1
            ),
            topic_id=DefaultTopicId()
        )

    def _parse_response(self, content: str) -> tuple[str, str]:
        """解析LLM响应"""
        # 提取explanation
        explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?=CODE:|```|$)', content, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        # 提取代码块
        code_match = re.search(r'```(?:\w+)?\s*\n(.+?)\n```', content, re.DOTALL)
        code = code_match.group(1).strip() if code_match else content

        return explanation, code

@default_subscription
class ReviewerAgent(RoutedAgent):
    """审查智能体 - 审查代码质量"""

    def __init__(self, model_client: ChatCompletionClient):
        super().__init__("Code reviewer")
        self._model_client = model_client

    @message_handler
    async def handle_code_submission(
        self,
        message: CodeSubmission,
        ctx: MessageContext
    ) -> None:
        """审查代码"""
        print(f"[Reviewer] Reviewing code (iteration {message.iteration})")

        # 构建审查提示
        review_prompt = f"""
Review the following code:

CODE:
{message.code}

EXPLANATION:
{message.explanation}

Evaluate the code on:
1. Correctness - Does it work?
2. Code quality - Is it clean and maintainable?
3. Best practices - Does it follow conventions?
4. Efficiency - Is it performant?
5. Documentation - Is it well-documented?

Provide your review in JSON format:
{{
    "correctness_score": <1-10>,
    "quality_score": <1-10>,
    "practices_score": <1-10>,
    "efficiency_score": <1-10>,
    "documentation_score": <1-10>,
    "overall_score": <1-10>,
    "approved": <true if overall_score >= 8, false otherwise>,
    "feedback": "<summary of your review>",
    "issues": ["<issue 1>", "<issue 2>", ...]
}}
"""

        # 生成审查
        response = await self._model_client.create(
            messages=[
                SystemMessage(content="You are a meticulous code reviewer."),
                UserMessage(content=review_prompt, source="system")
            ],
            json_output=True,
            cancellation_token=ctx.cancellation_token
        )

        # 解析审查结果
        review_data = json.loads(response.content)
        approved = review_data["approved"]

        print(f"[Reviewer] Review completed - "
              f"Score: {review_data['overall_score']}/10 - "
              f"{'APPROVED ✓' if approved else 'NEEDS REVISION ✗'}")

        if approved:
            # 代码通过审查
            await self.publish_message(
                FinalCode(
                    request_id=message.request_id,
                    code=message.code,
                    iterations=message.iteration + 1,
                    final_review=review_data["feedback"]
                ),
                topic_id=DefaultTopicId()
            )

        # 发送反馈（无论是否通过）
        await self.publish_message(
            ReviewFeedback(
                request_id=message.request_id,
                approved=approved,
                feedback=review_data["feedback"],
                issues=review_data.get("issues", []),
                iteration=message.iteration
            ),
            topic_id=DefaultTopicId()
        )

# ============================================================================
# 主程序
# ============================================================================

async def main():
    """运行代码审查系统"""

    # 创建运行时
    runtime = SingleThreadedAgentRuntime()

    # 创建模型客户端
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        # api_key="your-api-key"  # 如果需要
    )

    # 注册智能体
    await CoordinatorAgent.register(
        runtime,
        "CoordinatorAgent",
        lambda: CoordinatorAgent()
    )

    await CoderAgent.register(
        runtime,
        "CoderAgent",
        lambda: CoderAgent(model_client=model_client, max_iterations=3)
    )

    await ReviewerAgent.register(
        runtime,
        "ReviewerAgent",
        lambda: ReviewerAgent(model_client=model_client)
    )

    # 启动运行时
    runtime.start()

    # 发送代码请求
    request_id = str(uuid.uuid4())
    await runtime.publish_message(
        CodeRequest(
            request_id=request_id,
            task="Write a function to calculate the factorial of a number",
            language="python",
            requirements=[
                "Handle edge cases (0, negative numbers)",
                "Include input validation",
                "Add proper error handling",
                "Include docstring and type hints"
            ]
        ),
        topic_id=DefaultTopicId()
    )

    # 等待完成
    await runtime.stop_when_idle()

    # 清理
    await model_client.close()
    await runtime.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### 7.2 示例：客户服务系统（Handoff Pattern）

```python
# ============================================================================
# 客户服务多智能体系统 - 演示Handoff Pattern
# ============================================================================

import asyncio
import uuid
from dataclasses import dataclass
from typing import List

from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionCall,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ============================================================================
# 消息协议
# ============================================================================

@dataclass
class CustomerQuery:
    """客户查询"""
    session_id: str
    query: str

@dataclass
class ConversationMessage:
    """对话消息"""
    session_id: str
    context: List[LLMMessage]
    current_agent: str

@dataclass
class ConversationComplete:
    """对话完成"""
    session_id: str
    resolution: str

# ============================================================================
# 委托工具定义
# ============================================================================

def transfer_to_sales() -> str:
    """Transfer to sales department"""
    return "SalesAgent"

def transfer_to_support() -> str:
    """Transfer to technical support"""
    return "SupportAgent"

def transfer_to_billing() -> str:
    """Transfer to billing department"""
    return "BillingAgent"

def transfer_to_human() -> str:
    """Escalate to human agent"""
    return "HumanAgent"

# 创建工具
sales_tool = FunctionTool(
    transfer_to_sales,
    description="Transfer to sales for purchasing inquiries"
)

support_tool = FunctionTool(
    transfer_to_support,
    description="Transfer to support for technical issues"
)

billing_tool = FunctionTool(
    transfer_to_billing,
    description="Transfer to billing for payment issues"
)

human_tool = FunctionTool(
    transfer_to_human,
    description="Escalate to human agent for complex issues"
)

# ============================================================================
# 智能体实现
# ============================================================================

class HandoffCapableAgent(RoutedAgent):
    """支持Handoff的基础智能体"""

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        model_client: ChatCompletionClient,
        handoff_tools: List[FunctionTool],
        agent_topic_type: str
    ):
        super().__init__(description)
        self._name = name
        self._system_prompt = system_prompt
        self._model_client = model_client
        self._handoff_tools = {t.name: t for t in handoff_tools}
        self._agent_topic_type = agent_topic_type

    @message_handler
    async def handle_conversation(
        self,
        message: ConversationMessage,
        ctx: MessageContext
    ) -> None:
        """处理对话"""
        print(f"\n[{self._name}] Handling conversation")

        # 调用LLM
        response = await self._model_client.create(
            messages=[
                SystemMessage(content=self._system_prompt),
                *message.context
            ],
            tools=[t.schema for t in self._handoff_tools.values()],
            cancellation_token=ctx.cancellation_token
        )

        print(f"[{self._name}] Response: {response.content}")

        # 检查是否是工具调用（handoff）
        if isinstance(response.content, list) and \
           all(isinstance(c, FunctionCall) for c in response.content):
            # 处理handoff
            for call in response.content:
                if call.name in self._handoff_tools:
                    # 获取目标智能体类型
                    tool = self._handoff_tools[call.name]
                    import json
                    target_agent = await tool.run_json(
                        json.loads(call.arguments),
                        ctx.cancellation_token
                    )

                    print(f"[{self._name}] Handing off to {target_agent}")

                    # 创建新的上下文（包含handoff信息）
                    new_context = message.context + [
                        AssistantMessage(
                            content=f"Transferring you to {target_agent}",
                            source=self._name
                        )
                    ]

                    # 发布到目标智能体
                    await self.publish_message(
                        ConversationMessage(
                            session_id=message.session_id,
                            context=new_context,
                            current_agent=target_agent
                        ),
                        topic_id=TopicId(type=target_agent, source=message.session_id)
                    )
                    return

        # 文本响应 - 继续对话
        assert isinstance(response.content, str)

        # 更新上下文
        new_context = message.context + [
            AssistantMessage(content=response.content, source=self._name)
        ]

        # 模拟用户输入（实际应该从UI获取）
        user_input = input(f"\nYou: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            # 结束对话
            await self.publish_message(
                ConversationComplete(
                    session_id=message.session_id,
                    resolution=f"Conversation ended by user after interacting with {self._name}"
                ),
                topic_id=TopicId(type="Coordinator", source=message.session_id)
            )
            return

        # 继续对话
        new_context.append(UserMessage(content=user_input, source="user"))

        await self.publish_message(
            ConversationMessage(
                session_id=message.session_id,
                context=new_context,
                current_agent=self._agent_topic_type
            ),
            topic_id=TopicId(type=self._agent_topic_type, source=message.session_id)
        )

class CoordinatorAgent(RoutedAgent):
    """协调器智能体"""

    @message_handler
    async def handle_customer_query(
        self,
        message: CustomerQuery,
        ctx: MessageContext
    ) -> None:
        """处理客户查询"""
        print(f"\n{'='*60}")
        print(f"New customer query: {message.query}")
        print(f"{'='*60}")

        # 启动对话，从TriageAgent开始
        await self.publish_message(
            ConversationMessage(
                session_id=message.session_id,
                context=[UserMessage(content=message.query, source="user")],
                current_agent="TriageAgent"
            ),
            topic_id=TopicId(type="TriageAgent", source=message.session_id)
        )

    @message_handler
    async def handle_conversation_complete(
        self,
        message: ConversationComplete,
        ctx: MessageContext
    ) -> None:
        """处理对话完成"""
        print(f"\n{'='*60}")
        print(f"Conversation complete!")
        print(f"Resolution: {message.resolution}")
        print(f"{'='*60}\n")

# ============================================================================
# 主程序
# ============================================================================

async def main():
    """运行客户服务系统"""

    # 创建运行时
    runtime = SingleThreadedAgentRuntime()

    # 创建模型客户端
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini"
    )

    # 注册Coordinator
    await CoordinatorAgent.register(
        runtime,
        "Coordinator",
        lambda: CoordinatorAgent("Conversation coordinator")
    )

    # 注册TriageAgent
    await HandoffCapableAgent.register(
        runtime,
        "TriageAgent",
        lambda: HandoffCapableAgent(
            name="Triage Agent",
            description="Initial contact point",
            system_prompt="""You are a triage agent for ACME Inc.
Greet customers warmly and understand their needs.
Route them to:
- Sales: for purchasing inquiries
- Support: for technical issues
- Billing: for payment/invoice issues
- Human: for complex issues requiring human attention

Be concise and friendly.""",
            model_client=model_client,
            handoff_tools=[sales_tool, support_tool, billing_tool, human_tool],
            agent_topic_type="TriageAgent"
        )
    )

    # 注册SalesAgent
    await HandoffCapableAgent.register(
        runtime,
        "SalesAgent",
        lambda: HandoffCapableAgent(
            name="Sales Agent",
            description="Handles sales inquiries",
            system_prompt="""You are a sales agent for ACME Inc.
Help customers with product information and purchasing.
Be enthusiastic and persuasive.
If the customer needs technical help or has billing issues, transfer them.""",
            model_client=model_client,
            handoff_tools=[support_tool, billing_tool, human_tool],
            agent_topic_type="SalesAgent"
        )
    )

    # 注册SupportAgent
    await HandoffCapableAgent.register(
        runtime,
        "SupportAgent",
        lambda: HandoffCapableAgent(
            name="Support Agent",
            description="Handles technical issues",
            system_prompt="""You are a technical support agent for ACME Inc.
Help customers troubleshoot technical issues.
Be patient and methodical.
If you can't resolve the issue, escalate to human.""",
            model_client=model_client,
            handoff_tools=[sales_tool, billing_tool, human_tool],
            agent_topic_type="SupportAgent"
        )
    )

    # 注册BillingAgent
    await HandoffCapableAgent.register(
        runtime,
        "BillingAgent",
        lambda: HandoffCapableAgent(
            name="Billing Agent",
            description="Handles billing inquiries",
            system_prompt="""You are a billing agent for ACME Inc.
Help customers with invoices, payments, and refunds.
Be clear about policies and procedures.""",
            model_client=model_client,
            handoff_tools=[sales_tool, support_tool, human_tool],
            agent_topic_type="BillingAgent"
        )
    )

    # 添加订阅
    await runtime.add_subscription(
        TypeSubscription(topic_type="Coordinator", agent_type="Coordinator")
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="TriageAgent", agent_type="TriageAgent")
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="SalesAgent", agent_type="SalesAgent")
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="SupportAgent", agent_type="SupportAgent")
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="BillingAgent", agent_type="BillingAgent")
    )

    # 启动运行时
    runtime.start()

    # 发送客户查询
    session_id = str(uuid.uuid4())
    await runtime.publish_message(
        CustomerQuery(
            session_id=session_id,
            query="I want to buy your product but I'm having issues with payment"
        ),
        topic_id=TopicId(type="Coordinator", source=session_id)
    )

    # 等待完成
    await runtime.stop_when_idle()

    # 清理
    await model_client.close()
    await runtime.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 8. 最佳实践

### 8.1 消息设计

**✓ 好的做法：**

```python
@dataclass
class WellDesignedMessage:
    # 1. 包含唯一标识符
    message_id: str

    # 2. 包含时间戳
    timestamp: float

    # 3. 包含关联信息
    session_id: str
    correlation_id: str

    # 4. 类型明确的字段
    status: Literal["pending", "processing", "completed"]
    priority: int

    # 5. 完整的上下文
    context_data: dict

    # 6. 可选字段有默认值
    retry_count: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
```

**✗ 避免的做法：**

```python
@dataclass
class PoorlyDesignedMessage:
    # ✗ 字段太少，缺乏上下文
    data: str

    # ✗ 使用泛型类型
    stuff: dict
    thing: Any

    # ✗ 没有标识符
    # ✗ 没有时间戳
```

### 8.2 错误处理

```python
class RobustAgent(RoutedAgent):
    """健壮的智能体实现"""

    @message_handler
    async def handle_message(
        self,
        message: SomeMessage,
        ctx: MessageContext
    ) -> None:
        try:
            # 验证输入
            self._validate_message(message)

            # 处理消息
            result = await self._process_message(message, ctx)

            # 发送成功响应
            await self.publish_message(
                SuccessResponse(result=result),
                topic_id=...
            )

        except ValidationError as e:
            # 处理验证错误
            print(f"[{self.id.type}] Validation error: {e}")
            await self.publish_message(
                ErrorResponse(error_type="validation", message=str(e)),
                topic_id=...
            )

        except TimeoutError as e:
            # 处理超时
            print(f"[{self.id.type}] Timeout: {e}")
            # 可能需要重试
            await self._schedule_retry(message)

        except Exception as e:
            # 处理未预期的错误
            print(f"[{self.id.type}] Unexpected error: {e}")
            import traceback
            traceback.print_exc()

            await self.publish_message(
                ErrorResponse(error_type="internal", message="Internal error"),
                topic_id=...
            )

    def _validate_message(self, message: SomeMessage) -> None:
        """验证消息"""
        if not message.required_field:
            raise ValidationError("required_field is missing")
```

### 8.3 日志和监控

```python
import logging
from datetime import datetime

class ObservableAgent(RoutedAgent):
    """可观测的智能体"""

    def __init__(self, description: str):
        super().__init__(description)
        self._logger = logging.getLogger(f"agent.{self.id.type}")
        self._metrics = {
            "messages_received": 0,
            "messages_sent": 0,
            "errors": 0
        }

    @message_handler
    async def handle_message(
        self,
        message: SomeMessage,
        ctx: MessageContext
    ) -> None:
        # 记录接收
        self._metrics["messages_received"] += 1
        self._logger.info(
            f"Received {type(message).__name__}",
            extra={
                "message_type": type(message).__name__,
                "session_id": getattr(message, "session_id", None),
                "timestamp": datetime.now().isoformat()
            }
        )

        start_time = asyncio.get_event_loop().time()

        try:
            # 处理消息
            result = await self._process_message(message, ctx)

            # 记录成功
            elapsed = asyncio.get_event_loop().time() - start_time
            self._logger.info(
                f"Processed successfully in {elapsed:.3f}s",
                extra={
                    "duration_seconds": elapsed,
                    "status": "success"
                }
            )

        except Exception as e:
            # 记录错误
            self._metrics["errors"] += 1
            self._logger.error(
                f"Processing failed: {e}",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                exc_info=True
            )
            raise

    async def get_metrics(self) -> dict:
        """获取指标"""
        return self._metrics.copy()
```

### 8.4 测试策略

**单元测试智能体：**

```python
import pytest
from autogen_core import SingleThreadedAgentRuntime, TopicId

@pytest.mark.asyncio
async def test_agent_handles_message():
    """测试智能体处理消息"""
    # 创建运行时
    runtime = SingleThreadedAgentRuntime()

    # 创建mock model client
    mock_client = MockChatCompletionClient(
        responses=["This is a test response"]
    )

    # 注册智能体
    await MyAgent.register(
        runtime,
        "MyAgent",
        lambda: MyAgent(model_client=mock_client)
    )

    # 创建测试消息接收器
    received_messages = []

    @default_subscription
    class TestReceiver(RoutedAgent):
        @message_handler
        async def handle_response(self, message: Response, ctx: MessageContext):
            received_messages.append(message)

    await TestReceiver.register(runtime, "TestReceiver", lambda: TestReceiver("Test"))

    # 启动运行时
    runtime.start()

    # 发送测试消息
    await runtime.publish_message(
        TestMessage(content="test"),
        topic_id=TopicId("default", "default")
    )

    # 等待处理
    await runtime.stop_when_idle()

    # 断言
    assert len(received_messages) == 1
    assert received_messages[0].content == "This is a test response"

    # 清理
    await runtime.close()
```

**集成测试：**

```python
@pytest.mark.asyncio
async def test_multi_agent_workflow():
    """测试多智能体工作流"""
    runtime = SingleThreadedAgentRuntime()
    model_client = MockChatCompletionClient(...)

    # 注册所有智能体
    await AgentA.register(runtime, "AgentA", lambda: AgentA(...))
    await AgentB.register(runtime, "AgentB", lambda: AgentB(...))
    await AgentC.register(runtime, "AgentC", lambda: AgentC(...))

    # 设置订阅
    await runtime.add_subscription(...)

    runtime.start()

    # 触发工作流
    await runtime.publish_message(InitialMessage(...), ...)

    # 等待完成
    await runtime.stop_when_idle()

    # 验证最终状态
    # ...
```

### 8.5 性能优化

**1. 批处理消息**

```python
class BatchProcessingAgent(RoutedAgent):
    """批处理智能体"""

    def __init__(self, batch_size: int = 10, batch_timeout: float = 5.0):
        super().__init__("Batch processor")
        self._batch_size = batch_size
        self._batch_timeout = batch_timeout
        self._batch: List[Message] = []
        self._batch_task: Optional[asyncio.Task] = None

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        """累积消息并批处理"""
        self._batch.append(message)

        if len(self._batch) >= self._batch_size:
            # 达到批大小，立即处理
            await self._process_batch(ctx)
        elif self._batch_task is None:
            # 启动超时任务
            self._batch_task = asyncio.create_task(
                self._batch_timeout_handler(ctx)
            )

    async def _batch_timeout_handler(self, ctx: MessageContext) -> None:
        """批处理超时处理器"""
        await asyncio.sleep(self._batch_timeout)
        if self._batch:
            await self._process_batch(ctx)

    async def _process_batch(self, ctx: MessageContext) -> None:
        """处理批次"""
        batch = self._batch.copy()
        self._batch.clear()

        if self._batch_task:
            self._batch_task.cancel()
            self._batch_task = None

        # 批量处理
        results = await self._process_messages_in_batch(batch, ctx)

        # 发送结果
        for result in results:
            await self.publish_message(result, ...)
```

**2. 并发处理**

```python
class ConcurrentAgent(RoutedAgent):
    """支持并发处理的智能体"""

    def __init__(self, max_concurrent: int = 5):
        super().__init__("Concurrent processor")
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        """并发处理消息"""
        async with self._semaphore:
            # 限制并发数
            result = await self._process_message(message, ctx)
            await self.publish_message(result, ...)
```

---

## 9. 常见问题与解决方案

### 9.1 智能体没有收到消息

**问题：** 发布了消息但智能体没有收到

**可能原因和解决方案：**

1. **订阅没有正确设置**
```python
# 确保添加了订阅
await runtime.add_subscription(
    TypeSubscription(
        topic_type="YourTopicType",  # 必须匹配发布时的topic type
        agent_type="YourAgentType"   # 必须匹配注册时的agent type
    )
)
```

2. **Topic ID不匹配**
```python
# 发布消息
await runtime.publish_message(
    message,
    topic_id=TopicId(type="TaskTopic", source="session_123")
)

# Type-based subscription会创建Agent ID: (agent_type, "session_123")
# 确保使用相同的source
```

3. **运行时没有启动**
```python
runtime.start()  # 不要忘记启动！
```

### 9.2 消息循环和死锁

**问题：** 智能体之间形成消息循环

**解决方案：**

```python
@dataclass
class MessageWithTracking:
    """带追踪的消息"""
    content: str
    visited_agents: List[str]
    max_hops: int = 10

class SafeAgent(RoutedAgent):
    @message_handler
    async def handle_message(
        self,
        message: MessageWithTracking,
        ctx: MessageContext
    ) -> None:
        # 检查是否已访问
        if self.id.type in message.visited_agents:
            print(f"[{self.id.type}] Loop detected, stopping")
            return

        # 检查最大跳数
        if len(message.visited_agents) >= message.max_hops:
            print(f"[{self.id.type}] Max hops reached, stopping")
            return

        # 添加到已访问列表
        new_visited = message.visited_agents + [self.id.type]

        # 处理并转发
        result = await self._process(message.content)

        await self.publish_message(
            MessageWithTracking(
                content=result,
                visited_agents=new_visited,
                max_hops=message.max_hops
            ),
            topic_id=...
        )
```

### 9.3 状态不一致

**问题：** 多租户场景下状态混乱

**解决方案：**

```python
class StatefulAgent(RoutedAgent):
    """正确的状态管理"""

    def __init__(self):
        super().__init__("Stateful agent")
        # 使用字典隔离不同会话的状态
        self._sessions: Dict[str, SessionState] = {}

    @message_handler
    async def handle_message(
        self,
        message: SessionMessage,
        ctx: MessageContext
    ) -> None:
        # 获取或创建会话状态
        if message.session_id not in self._sessions:
            self._sessions[message.session_id] = SessionState()

        state = self._sessions[message.session_id]

        # 使用会话特定的状态
        await self._process_with_state(message, state, ctx)

    async def cleanup_old_sessions(self, max_age: float) -> None:
        """清理旧会话"""
        current_time = asyncio.get_event_loop().time()
        expired = [
            sid for sid, state in self._sessions.items()
            if current_time - state.last_activity > max_age
        ]
        for sid in expired:
            del self._sessions[sid]
```

### 9.4 调试技巧

**启用详细日志：**

```python
import logging

# 设置AutoGen日志级别
logging.basicConfig(level=logging.WARNING)
logging.getLogger("autogen_core").setLevel(logging.DEBUG)

# 这会显示：
# - 消息发布
# - 消息处理
# - 智能体创建
# - 订阅信息
```

**添加调试中间件：**

```python
class DebugAgent(RoutedAgent):
    """带调试功能的智能体"""

    @message_handler
    async def handle_any_message(
        self,
        message: Any,
        ctx: MessageContext
    ) -> None:
        # 打印所有收到的消息
        print(f"\n{'='*60}")
        print(f"[DEBUG] Agent: {self.id}")
        print(f"[DEBUG] Message Type: {type(message).__name__}")
        print(f"[DEBUG] Message Content: {message}")
        print(f"[DEBUG] Context: {ctx}")
        print(f"{'='*60}\n")

        # 继续处理...
```

---

## 总结

这份指南涵盖了使用AutoGen构建多智能体协调系统的所有关键方面：

1. **核心概念**: Agent, AgentID, Topic, Subscription
2. **消息协议**: 如何设计清晰的消息类型
3. **智能体实现**: 从基础到高级的智能体开发
4. **通信模式**: 直接消息、广播、订阅
5. **设计模式**: Handoff, Reflection, Group Chat
6. **运行时环境**: 单线程和分布式运行时
7. **完整示例**: 可运行的端到端系统
8. **最佳实践**: 错误处理、日志、测试、性能
9. **问题排查**: 常见问题和解决方案

使用这份指南，任何大型语言模型或开发者都应该能够理解并实现一个生产级别的多智能体系统。

**下一步建议：**
1. 从简单的两个智能体开始练习
2. 逐步增加复杂度
3. 实现一个真实的业务场景
4. 添加监控和日志
5. 进行性能优化
6. 考虑分布式部署
