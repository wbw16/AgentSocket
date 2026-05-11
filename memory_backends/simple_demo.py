from __future__ import annotations

from dataclasses import asdict, dataclass, field

from agent.core.types import ToolCallRecord


@dataclass(slots=True)
class SimpleDemoMemory:
    """Tiny memory backend for smoke-testing memory replacement."""

    _messages: list[dict[str, str]] = field(default_factory=list)
    _action_history: list[ToolCallRecord] = field(default_factory=list)

    def append_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def append_action(self, record: ToolCallRecord) -> None:
        self._action_history.append(record)

    def messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def action_history(self) -> list[ToolCallRecord]:
        return list(self._action_history)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        return self.messages()[-k:]

    def summarize(self, max_tokens: int) -> str:
        text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in self._messages)
        return text[:max_tokens]

    def snapshot(self) -> dict:
        return {
            "message_count": len(self._messages),
            "action_count": len(self._action_history),
            "messages": self.messages(),
            "action_history": [asdict(record) for record in self._action_history],
        }
