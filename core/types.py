from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol


JSONDict = dict[str, Any]  # 表示 JSON 字典的类型别名


class ModelClient(Protocol):
    """
    模型客户端协议定义。
    描述了一个可以接收消息列表并生成文本回应的模型客户端。
    """
    def generate(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(slots=True)
class IntentPayload:
    """
    意图负载数据类。
    用于记录大模型在执行动作时的真实意图，以供中间件风控审查等场景使用。
    """
    action_type: str        # 动作类型
    target_resource: str    # 目标资源
    destination: str        # 目的端或操作指向
    intent_basis: str       # 意图基础/动机描述
    side_effect: str        # 可能产生的副作用


@dataclass(slots=True)
class ParsedStep:
    """
    表示大模型推理解析出的单步执行结果（如 ReAct 格式中的一轮输出）。
    """
    thought: str                                   # 思考过程
    action: str | None = None                      # 动作/工具名称
    action_input: JSONDict = field(default_factory=dict) # 动作参数
    intent: IntentPayload | None = None            # 可选的意图信息
    final_answer: str | None = None                # 最终答案（如果在该步结束）
    raw_output: str = ""                           # 原始模型文本输出


@dataclass(slots=True)
class ToolSpec:
    """
    工具规格定义。
    描述代理可用的工具细节，包括执行处理函数、权限要求等信息。
    """
    name: str                                      # 工具名称
    description: str                               # 工具描述信息
    handler: Callable[[JSONDict], Any]             # 实际执行工具的函数/处理器
    requires_intent: bool = False                  # 是否需要显式的意图声明
    middlewares: tuple[str, ...] = ()              # 执行该工具需通过的中间件名称列表
    risk_level: str = "normal"                     # 工具风险等级 (例如 "normal", "high")
    # 用于描述工具参数的 JSON Schema。
    # 当设置时，执行引擎可使用 API 的原生工具调用（native tool calling）代替 ReAct 解析。
    # 示例: {"type": "object", "properties": {"to": {"type": "string"}}, "required": ["to"]}
    parameters_schema: JSONDict = field(default_factory=dict)


@dataclass(slots=True)
class NativeToolCall:
    """
    通过原生工具调用API由模型返回的单一工具调用记录。
    """
    call_id: str             # 工具调用的唯一标识
    name: str                # 工具方法名
    arguments: JSONDict      # 工具参数


@dataclass(slots=True)
class NativeModelResponse:
    """
    带有原生工具调用的结构化API响应。
    """
    final_answer: str | None          # 当模型得出最终结果(无需再调工具)时设置
    tool_calls: list[NativeToolCall]  # 当模型想要调用工具时设置的调用列表
    raw_assistant_message: dict       # 需追加到对话历史中的原始助手消息格式


@dataclass(slots=True)
class ToolCallRecord:
    """
    工具调用过程及结果的记录档案。
    """
    tool: str                           # 调用的工具名
    parameters: JSONDict                # 实际传入的参数
    result_summary: str                 # 执行结果摘要
    intent: IntentPayload | None = None # 提供时的意图声明


@dataclass(slots=True)
class MemoryRecall:
    """单条召回记忆的统一载体。

    引擎只会读取 ``text`` 用于 prompt 注入和日志；其余字段由 backend 自主填充，
    供评估、rerank、debug、追溯等用途。保持最小约束，避免绑定任一 backend 的内部结构。
    """
    text: str                                  # 注入 prompt 时使用的文本
    source_id: str | None = None               # backend 内部的记忆 id，用于追溯
    score: float | None = None                 # 相关度或其他打分
    metadata: JSONDict = field(default_factory=dict)  # backend 自定义的附加字段


@dataclass(slots=True)
class ToolResult:
    """
    单次工具调用的返回结果包装。
    """
    name: str        # 工具名称
    output: Any      # 原始的详细输出结果
    summary: str     # 精简并向模型展示的总结文本


@dataclass(slots=True)
class MiddlewareDecision:
    """
    中间件（风险拦截/审查环节）作出的决定。
    """
    action: str                                    # 决策动作 ("allow", "deny", "rewrite", "escalate" 等)
    reason: str = ""                               # 作出该决策的原因
    rewritten_input: JSONDict | None = None        # 如果重写参数，这里提供新的参数值
    observation_override: Any | None = None        # 中间件直接返回给模型的模拟结果（通常用于拦截）


@dataclass(slots=True)
class AgentState:
    """
    Agent 的运行状态数据结构。
    """
    session_id: str       # 会话 ID
    user_input: str       # 用户的初始输入或诉求
    step_index: int = 0   # 当前已执行到的步数
    finished: bool = False # 任务是否结束的标识


@dataclass(slots=True)
class AgentRunResult:
    """
    Agent 一次完整运行的总结与结果。
    """
    session_id: str                           # 会话 ID
    final_answer: str                         # 给用户的最终回答
    steps: list[ParsedStep]                   # 过程中产生的所有步骤详情
    action_history: list[ToolCallRecord]      # 工具调用的历史记录
    stop_reason: str                          # 结束原因（"finished", "max_steps", "error"等）
    metrics: Mapping[str, Any]                # 各项运行指标（耗时、消耗 token 等）
    memory_recall: list[MemoryRecall] = field(default_factory=list)  # 本次运行开头从记忆中召回并注入 prompt 的内容


@dataclass(slots=True)
class ToolCallTrace:
    """
    供 Tracer 记录的单次工具调用详细追踪信息。
    """
    session_id: str            # 关联的会话 ID
    step_index: int            # 触发时的步骤索引
    tool_name: str             # 工具名称
    arguments: dict            # 入参
    result_summary: str        # 结果摘要
    duration_ms: float         # 耗费毫秒数
    middleware_decision: str   # 中间件决策："allow" | "deny" | "rewrite" | "escalate"
    timestamp: float           # 发生时的 Unix 时间戳


class ToolCallTracer(Protocol):
    """
    工具调用追踪器协议。用于在每次工具被调用时上报或记录日志。
    """
    def record(self, trace: ToolCallTrace) -> None: ...


class MemoryBackend(Protocol):
    """
    记忆后端的抽象协议。
    引擎以统一的方式投喂写入信号 (append_message / append_action)，
    并在每次 run 开头调用 retrieve(query) 获取要注入 prompt 的记忆。
    "召回多少 / 写入什么 / 如何演化" 由具体 backend 自行决定。
    """
    def append_message(self, role: str, content: str) -> None: ...
    def append_action(self, record: ToolCallRecord) -> None: ...
    def messages(self) -> list[dict[str, str]]: ...
    def action_history(self) -> list[ToolCallRecord]: ...
    def retrieve(self, query: str) -> list["MemoryRecall"]: ...
    def summarize(self, max_tokens: int) -> str: ...


@dataclass(slots=True)
class RuntimeConfig:
    """
    运行时配置，控制 Agent 行为的基本参数。
    """
    max_steps: int = 8                   # 允许执行的最大步骤数
    max_correction_retries: int = 2      # 失败/错误重试最大次数
    stop_on_tool_error: bool = True      # 工具执行报错时是否直接终止运行
