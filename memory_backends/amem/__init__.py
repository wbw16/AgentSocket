from __future__ import annotations

"""A-MEM 记忆后端包。

快速使用:
    from agent.memory_backends.amem import AMemBackend, AMemConfig
    from agent.memory_backends.amem import AnthropicAdapter

    llm = AnthropicAdapter(anthropic_client)
    memory = AMemBackend(config=AMemConfig(enable_evolution=False), llm=llm)
"""

from agent.memory_backends.amem.backend import AMemBackend
from agent.memory_backends.amem.config import AMemConfig
from agent.memory_backends.amem.llm import AnthropicAdapter, LLMClient, OpenAIAdapter
from agent.memory_backends.amem.embedder import Embedder, SentenceTransformerEmbedder
from agent.memory_backends.amem.store import ChromaStore, InMemoryStore, VectorStore

__all__ = [
    "AMemBackend",
    "AMemConfig",
    # LLM 适配层
    "LLMClient",
    "AnthropicAdapter",
    "OpenAIAdapter",
    # Embedder
    "Embedder",
    "SentenceTransformerEmbedder",
    # Store
    "VectorStore",
    "InMemoryStore",
    "ChromaStore",
]
