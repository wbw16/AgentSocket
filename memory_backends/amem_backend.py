"""A-mem backend adapter.

Wraps AgenticMemorySystem as a MemoryBackend protocol.

fast_ingest=True (default): skips per-turn LLM analysis during append_message,
directly inserts into the retriever. Suitable for ingesting long conversation
histories without burning hundreds of LLM calls.

fast_ingest=False: uses add_note() which triggers LLM keyword/context extraction
and memory evolution on every turn. Slower but richer metadata.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

_AMEM_DIR = Path(__file__).parent.parent.parent / "A-mem"
if str(_AMEM_DIR) not in sys.path:
    sys.path.insert(0, str(_AMEM_DIR))

from memory_layer import AgenticMemorySystem, MemoryNote  # noqa: E402

from agent.core.types import MemoryRecall, ToolCallRecord  # noqa: E402


@dataclass
class AMEMBackend:
    """MemoryBackend backed by A-mem's AgenticMemorySystem.

    Parameters
    ----------
    llm_backend:  "openai" | "ollama" | "sglang"
    llm_model:    model name passed to A-mem's LLMController
    api_key:      API key (reads OPENAI_API_KEY from env if None)
    api_base:     base URL for OpenAI-compatible endpoints
    embed_model:  sentence-transformers model for retrieval
    retrieve_k:   number of memories to return per retrieve() call
    fast_ingest:  skip per-turn LLM analysis (recommended for long conversations)
    """

    llm_backend: str = "openai"
    llm_model: str = "gpt-4o-mini"
    api_key: str | None = None
    api_base: str | None = None
    embed_model: str = "all-MiniLM-L6-v2"
    retrieve_k: int = 10
    fast_ingest: bool = True

    _messages: list[dict[str, str]] = field(default_factory=list, init=False)
    _actions: list[ToolCallRecord] = field(default_factory=list, init=False)
    _mem: AgenticMemorySystem = field(init=False)

    def __post_init__(self) -> None:
        self._mem = AgenticMemorySystem(
            model_name=self.embed_model,
            llm_backend=self.llm_backend,
            llm_model=self.llm_model,
            api_key=self.api_key,
            api_base=self.api_base,
        )

    def append_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        text = f"{role}: {content}"
        if self.fast_ingest:
            self._fast_add(text, role)
        else:
            self._mem.add_note(text)

    def _fast_add(self, text: str, role: str) -> None:
        """Insert directly into memories + retriever, bypassing LLM analysis."""
        note = MemoryNote(
            content=text,
            keywords=[],
            context="conversation",
            tags=[role],
            llm_controller=None,
        )
        self._mem.memories[note.id] = note
        self._mem.retriever.add_documents([text])

    def append_action(self, record: ToolCallRecord) -> None:
        self._actions.append(record)

    def messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def action_history(self) -> list[ToolCallRecord]:
        return list(self._actions)

    def retrieve(self, query: str) -> list[MemoryRecall]:
        if not self._mem.memories:
            return []
        raw: str = self._mem.find_related_memories_raw(query, k=self.retrieve_k)
        if not raw:
            return []
        return [MemoryRecall(text=raw, source_id="amem", score=None)]

    def summarize(self, max_tokens: int) -> str:
        lines = [f"{m['role']}: {m['content']}" for m in self._messages]
        text = "\n".join(lines)
        return text[:max_tokens]
