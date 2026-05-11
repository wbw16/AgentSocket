from __future__ import annotations

from dataclasses import dataclass, field

from agent.core.types import ToolCallRecord


@dataclass(slots=True)
class InMemorySessionMemory:
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
        return []

    def summarize(self, max_tokens: int) -> str:
        return ""
