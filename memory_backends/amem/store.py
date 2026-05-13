from __future__ import annotations

from typing import Protocol


class VectorStore(Protocol):
    """向量存储协议。"""

    def add(self, note_id: str, vector: list[float], payload: dict) -> None: ...
    def search(self, query_vector: list[float], k: int) -> list[tuple[str, float, dict]]: ...
    def get(self, note_id: str) -> dict | None: ...
    def update(self, note_id: str, payload: dict) -> None: ...
    def delete(self, note_id: str) -> None: ...
    def count(self) -> int: ...


class InMemoryStore:
    """基于 numpy 余弦相似度的纯内存向量存储，无需外部依赖。"""

    def __init__(self) -> None:
        # note_id -> (vector, payload)
        self._data: dict[str, tuple[list[float], dict]] = {}

    def add(self, note_id: str, vector: list[float], payload: dict) -> None:
        self._data[note_id] = (vector, dict(payload))

    def search(self, query_vector: list[float], k: int) -> list[tuple[str, float, dict]]:
        if not self._data:
            return []
        import numpy as np

        q = np.array(query_vector, dtype=float)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        scores: list[tuple[str, float, dict]] = []
        for note_id, (vec, payload) in self._data.items():
            v = np.array(vec, dtype=float)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                sim = 0.0
            else:
                sim = float(np.dot(q, v) / (q_norm * v_norm))
            scores.append((note_id, sim, payload))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def get(self, note_id: str) -> dict | None:
        entry = self._data.get(note_id)
        return dict(entry[1]) if entry else None

    def update(self, note_id: str, payload: dict) -> None:
        if note_id in self._data:
            vec, old_payload = self._data[note_id]
            merged = {**old_payload, **payload}
            self._data[note_id] = (vec, merged)

    def delete(self, note_id: str) -> None:
        self._data.pop(note_id, None)

    def count(self) -> int:
        return len(self._data)


class ChromaStore:
    """基于 ChromaDB 的持久化向量存储。"""

    def __init__(self, persist_dir: str, collection_name: str = "amem") -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, note_id: str, vector: list[float], payload: dict) -> None:
        # ChromaDB 的 metadata 值只支持 str/int/float/bool，需要序列化 list
        meta = _serialize_payload(payload)
        self._col.add(ids=[note_id], embeddings=[vector], metadatas=[meta])

    def search(self, query_vector: list[float], k: int) -> list[tuple[str, float, dict]]:
        n = self._col.count()
        if n == 0:
            return []
        actual_k = min(k, n)
        results = self._col.query(
            query_embeddings=[query_vector],
            n_results=actual_k,
            include=["metadatas", "distances"],
        )
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        out = []
        for nid, dist, meta in zip(ids, distances, metadatas):
            # cosine space: distance = 1 - similarity
            score = 1.0 - dist
            out.append((nid, score, _deserialize_payload(meta)))
        return out

    def get(self, note_id: str) -> dict | None:
        result = self._col.get(ids=[note_id], include=["metadatas"])
        if not result["ids"]:
            return None
        return _deserialize_payload(result["metadatas"][0])

    def update(self, note_id: str, payload: dict) -> None:
        existing = self.get(note_id)
        if existing is None:
            return
        merged = {**existing, **payload}
        meta = _serialize_payload(merged)
        self._col.update(ids=[note_id], metadatas=[meta])

    def delete(self, note_id: str) -> None:
        self._col.delete(ids=[note_id])

    def count(self) -> int:
        return self._col.count()


# ---------------------------------------------------------------------------
# ChromaDB payload 序列化辅助（list -> JSON string）
# ---------------------------------------------------------------------------

import json as _json


def _serialize_payload(payload: dict) -> dict:
    """将 list 字段序列化为 JSON 字符串，以满足 ChromaDB metadata 限制。"""
    out = {}
    for k, v in payload.items():
        if isinstance(v, list):
            out[k] = _json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def _deserialize_payload(meta: dict) -> dict:
    """将 JSON 字符串字段反序列化回 list。"""
    out = {}
    for k, v in meta.items():
        if isinstance(v, str):
            try:
                parsed = _json.loads(v)
                out[k] = parsed if isinstance(parsed, list) else v
            except (_json.JSONDecodeError, ValueError):
                out[k] = v
        else:
            out[k] = v
    return out
