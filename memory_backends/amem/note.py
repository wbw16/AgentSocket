from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryNote:
    """单条记忆节点，存储内容及演化后的元数据。"""

    content: str                          # 原始文本内容
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    keywords: list[str] = field(default_factory=list)   # LLM 提取的关键词
    context: str = ""                     # LLM 生成的一句话摘要
    tags: list[str] = field(default_factory=list)        # 分类标签
    links: list[str] = field(default_factory=list)       # 关联的其他 note id
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_payload(self) -> dict:
        """序列化为 VectorStore payload 字典。"""
        return {
            "content": self.content,
            "keywords": self.keywords,
            "context": self.context,
            "tags": self.tags,
            "links": self.links,
            "created_at": self.created_at,
        }

    @classmethod
    def from_payload(cls, note_id: str, payload: dict) -> MemoryNote:
        """从 VectorStore payload 反序列化。"""
        return cls(
            id=note_id,
            content=payload.get("content", ""),
            keywords=payload.get("keywords", []),
            context=payload.get("context", ""),
            tags=payload.get("tags", []),
            links=payload.get("links", []),
            created_at=payload.get("created_at", ""),
        )
