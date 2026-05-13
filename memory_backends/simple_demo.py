from __future__ import annotations

from dataclasses import asdict, dataclass, field

from agent.core.types import MemoryRecall, ToolCallRecord


@dataclass(slots=True)
class SimpleDemoMemory:
    """一个微型简单的记忆存储后端，主要用于冒烟测试(smoke-testing)或演示如何覆盖和替换记忆模块。"""

    _messages: list[dict[str, str]] = field(default_factory=list)        # 保存消息队列的历史字典
    _action_history: list[ToolCallRecord] = field(default_factory=list)   # 存储使用过的动作工具纪录
    recall_k: int = 5                                                     # retrieve 默认返回最近 k 条

    def append_message(self, role: str, content: str) -> None:
        """追加一条系统或角色的对话消息到记录表中。"""
        self._messages.append({"role": role, "content": content})

    def append_action(self, record: ToolCallRecord) -> None:
        """追加一次产生过的工具执行足迹。"""
        self._action_history.append(record)

    def messages(self) -> list[dict[str, str]]:
        """获取所有对话消息历史，返回拷贝以防止遭外部意外修改。"""
        return list(self._messages)

    def action_history(self) -> list[ToolCallRecord]:
        """获得当前的动作列表"""
        return list(self._action_history)

    def retrieve(self, query: str) -> list[MemoryRecall]:
        """冒烟用的粗糙召回：返回最近 recall_k 条消息，包成 MemoryRecall。"""
        recent = self._messages[-self.recall_k:]
        return [
            MemoryRecall(
                text=f"{msg['role']}: {msg['content']}",
                source_id=f"msg:{idx}",
                metadata={"role": msg["role"]},
            )
            for idx, msg in enumerate(recent, start=max(0, len(self._messages) - self.recall_k))
        ]

    def summarize(self, max_tokens: int) -> str:
        """以最大 Token 大小限制来简单地按换行缝合聊天日志以作摘要。"""
        text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in self._messages)
        return text[:max_tokens]

    def snapshot(self) -> dict:
        """创建当时全部记忆相关的瞬间快照数据表，方便供日志导出器利用。"""
        return {
            "message_count": len(self._messages),
            "action_count": len(self._action_history),
            "messages": self.messages(),
            "action_history": [asdict(record) for record in self._action_history],
        }
