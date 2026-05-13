from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agent.memory_backends.amem.config import AMemConfig
from agent.memory_backends.amem.note import MemoryNote
from agent.core.types import MemoryRecall, ToolCallRecord

logger = logging.getLogger("amem.backend")


@dataclass
class AMemBackend:
    """A-MEM 记忆后端，实现 MemoryBackend Protocol。

    构造参数:
        config      : AMemConfig 实例（所有消融旋钮）
        llm         : LLMClient 实例（用于元数据提取和 evolution）
                      如果 config.enable_metadata_extraction 和 config.enable_evolution
                      都为 False，可以传 None。
        embedder    : Embedder 实例（默认由 config.embedding_model 自动构造）
        store       : VectorStore 实例（默认由 config.persist_dir 自动构造）
    """

    config: AMemConfig = field(default_factory=AMemConfig)
    llm: object | None = None          # LLMClient Protocol
    embedder: object | None = None     # Embedder Protocol（None 则自动构造）
    store: object | None = None        # VectorStore Protocol（None 则自动构造）

    # 原始对话历史（引擎喂给 model 的完整消息流）
    _raw_messages: list[dict] = field(default_factory=list, init=False)
    # 工具调用历史
    _action_history: list[ToolCallRecord] = field(default_factory=list, init=False)

    # 延迟初始化的内部组件
    _embedder: object = field(default=None, init=False, repr=False)
    _store: object = field(default=None, init=False, repr=False)
    _extractor: object = field(default=None, init=False, repr=False)
    _evolution: object = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._setup_embedder()
        self._setup_store()
        self._setup_extractor()
        self._setup_evolution()
        self._initialized = True

    def _setup_embedder(self) -> None:
        if self.embedder is not None:
            self._embedder = self.embedder
            return
        from agent.memory_backends.amem.embedder import SentenceTransformerEmbedder
        self._embedder = SentenceTransformerEmbedder(self.config.embedding_model)

    def _setup_store(self) -> None:
        if self.store is not None:
            self._store = self.store
            return
        if self.config.persist_dir is not None:
            from agent.memory_backends.amem.store import ChromaStore
            self._store = ChromaStore(
                persist_dir=self.config.persist_dir,
                collection_name=self.config.collection_name,
            )
        else:
            from agent.memory_backends.amem.store import InMemoryStore
            self._store = InMemoryStore()

    def _setup_extractor(self) -> None:
        if not self.config.enable_metadata_extraction or self.llm is None:
            self._extractor = None
            return
        from agent.memory_backends.amem.extractor import MetadataExtractor
        self._extractor = MetadataExtractor(self.llm, max_retries=self.config.llm_max_retries)

    def _setup_evolution(self) -> None:
        if not self.config.enable_evolution or self.llm is None:
            self._evolution = None
            return
        from agent.memory_backends.amem.evolution import EvolutionPolicy
        self._evolution = EvolutionPolicy(self.llm, max_retries=self.config.llm_max_retries)

    # ------------------------------------------------------------------
    # MemoryBackend Protocol
    # ------------------------------------------------------------------

    def append_message(self, role: str, content: str) -> None:
        """追加消息到原始历史，并按写入策略决定是否落库。"""
        self._raw_messages.append({"role": role, "content": content})

        should_write = (
            (role == "user" and self.config.write_user_message)
            or (role == "assistant" and self.config.write_assistant_message)
        )
        if not should_write:
            return

        self._ensure_initialized()
        self._write_note(content)

    def append_action(self, record: ToolCallRecord) -> None:
        """追加工具调用记录；按 write_tool_actions 决定是否落库。"""
        self._action_history.append(record)
        if not self.config.write_tool_actions:
            return
        self._ensure_initialized()
        summary = f"[tool:{record.tool}] {record.result_summary}"
        self._write_note(summary)

    def messages(self) -> list[dict]:
        return list(self._raw_messages)

    def action_history(self) -> list[ToolCallRecord]:
        return list(self._action_history)

    def retrieve(self, query: str) -> list[MemoryRecall]:
        """向量检索，返回最相关的 recall_k 条记忆。"""
        self._ensure_initialized()
        if self._store.count() == 0:
            return []

        vec = self._embedder.encode(query)
        hits = self._store.search(vec, k=self.config.recall_k)

        recalls: list[MemoryRecall] = []
        for note_id, score, payload in hits:
            text = self._build_inject_text(payload)
            recalls.append(MemoryRecall(
                text=text,
                source_id=note_id,
                score=score,
                metadata={
                    "keywords": payload.get("keywords", []),
                    "context": payload.get("context", ""),
                    "tags": payload.get("tags", []),
                },
            ))
        return recalls

    def summarize(self, max_tokens: int) -> str:
        """简单拼接最近消息作为摘要（占位实现）。"""
        text = "\n".join(f"{m['role']}: {m['content']}" for m in self._raw_messages)
        return text[:max_tokens]

    # ------------------------------------------------------------------
    # 内部写入流程
    # ------------------------------------------------------------------

    def _write_note(self, content: str) -> None:
        """创建 MemoryNote，提取元数据，写入 store，触发 evolution。"""
        note = MemoryNote(content=content)

        # 元数据提取
        if self._extractor is not None:
            meta = self._extractor.extract(content)
            note.keywords = meta.get("keywords", [])
            note.context  = meta.get("context", "")
            note.tags      = meta.get("tags", [])

        # 向量化并写入 store
        vec = self._embedder.encode(content)
        self._store.add(note.id, vec, note.to_payload())
        logger.debug("Stored note %s (store size=%d)", note.id, self._store.count())

        # Evolution
        if self._evolution is not None:
            neighbors = self._fetch_neighbors(note)
            if len(neighbors) >= self.config.evolution_threshold:
                updates = self._evolution.evolve(note, neighbors)
                self._apply_updates(updates)

    def _fetch_neighbors(self, note: MemoryNote) -> list[MemoryNote]:
        """检索 evolution 用的邻居（排除 note 自身）。"""
        vec = self._embedder.encode(note.content)
        # 多取一个以防自身出现在结果里
        hits = self._store.search(vec, k=self.config.evolution_neighbors_k + 1)
        neighbors: list[MemoryNote] = []
        for nid, _score, payload in hits:
            if nid == note.id:
                continue
            neighbors.append(MemoryNote.from_payload(nid, payload))
            if len(neighbors) >= self.config.evolution_neighbors_k:
                break
        return neighbors

    def _apply_updates(self, updates: list) -> None:
        """把 evolution 返回的 NoteUpdate 列表应用到 store。"""
        for upd in updates:
            self._store.update(upd.note_id, upd.patch)
            logger.debug("Evolution updated note %s: %s", upd.note_id, list(upd.patch.keys()))

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _build_inject_text(self, payload: dict) -> str:
        """按 config.inject_field 决定注入 prompt 的文本格式。"""
        if self.config.inject_field == "text":
            return payload.get("content", "")
        # "content+keywords+context" 模式
        parts = [payload.get("content", "")]
        kw = payload.get("keywords", [])
        if kw:
            parts.append("keywords: " + ", ".join(kw))
        ctx = payload.get("context", "")
        if ctx:
            parts.append("context: " + ctx)
        return " | ".join(p for p in parts if p)
