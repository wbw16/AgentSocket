from __future__ import annotations

from dataclasses import dataclass, field

from agent.core.types import ToolCallRecord


@dataclass(slots=True)
class InMemorySessionMemory:
    """
    内存中的会话记忆实现类。
    用于在纯内存中记录对话消息和工具执行历史，适用于不需要持久化的单次运行或测试环境。
    """
    _messages: list[dict[str, str]] = field(default_factory=list)        # 存储对话历史
    _action_history: list[ToolCallRecord] = field(default_factory=list)   # 存储工具调用历史

    def append_message(self, role: str, content: str) -> None:
        """
        向记忆中追加一条新的对话消息。
        """
        self._messages.append({"role": role, "content": content})

    def append_action(self, record: ToolCallRecord) -> None:
        """
        向记忆中追加一条工具执行记录。
        """
        self._action_history.append(record)

    def messages(self) -> list[dict[str, str]]:
        """
        返回所有已保存的对话消息列表（为避免外部直接修改原列表，返回一个浅拷贝）。
        """
        return list(self._messages)

    def action_history(self) -> list[ToolCallRecord]:
        """
        返回所有已保存的工具调用历史（为避免外部直接修改原列表，返回一个浅拷贝）。
        """
        return list(self._action_history)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        语义检索接口占位符，返回与查询最相关的前 k 条记忆。
        (基础实现暂时为空)。
        """
        return []

    def summarize(self, max_tokens: int) -> str:
        """
        记忆摘要接口占位符，根据最大Token限制生成对话摘要。
        (基础实现暂时为空)。
        """
        return ""
