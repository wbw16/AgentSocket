from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AMemConfig:
    """AMemBackend 的所有配置旋钮，每个字段对应一个消融实验维度。"""

    # ---- 召回 ----
    recall_k: int = 5
    # "text" | "content+keywords+context" — 控制注入 prompt 的字段组合
    inject_field: str = "text"

    # ---- 写入策略 ----
    write_user_message: bool = True
    write_assistant_message: bool = True
    write_tool_actions: bool = False

    # ---- 元数据提取 ----
    enable_metadata_extraction: bool = True

    # ---- 演化 ----
    enable_evolution: bool = True
    # 新 note 至少有 N 个邻居才触发 evolution
    evolution_threshold: int = 3
    # evolution 考察多少个邻居
    evolution_neighbors_k: int = 5

    # ---- 存储后端 ----
    # None 则使用 InMemoryStore；设置路径则使用 ChromaStore 持久化
    persist_dir: str | None = None
    collection_name: str = "amem"
    embedding_model: str = "all-MiniLM-L6-v2"

    # ---- 健壮性 ----
    llm_max_retries: int = 2
