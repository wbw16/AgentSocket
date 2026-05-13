from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """向量化协议。"""

    def encode(self, text: str) -> list[float]:
        """将文本编码为向量。"""
        ...

    def dim(self) -> int:
        """返回向量维度。"""
        ...


class SentenceTransformerEmbedder:
    """使用 sentence-transformers 做本地向量化。"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        # 缓存维度，避免重复推断
        # get_embedding_dimension 是新名称，旧版本用 get_sentence_embedding_dimension
        get_dim = getattr(self._model, "get_embedding_dimension", None) or \
                  getattr(self._model, "get_sentence_embedding_dimension", None)
        self._dim: int = get_dim()

    def encode(self, text: str) -> list[float]:
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def dim(self) -> int:
        return self._dim
