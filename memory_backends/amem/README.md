# A-MEM 记忆后端

AgentSocket 的 A-MEM (Agentic Memory) 后端实现。基于向量检索 + LLM 元数据提取 + 记忆演化三层机制。

## 安装依赖

```bash
pip install sentence-transformers chromadb numpy
```

## 快速使用

```python
from agent.memory_backends.amem import AMemBackend, AMemConfig, AnthropicAdapter
from agent.clients import AnthropicClient
import agent

# 构造 LLM 适配器
client = AnthropicClient()
llm = AnthropicAdapter(client)

# 构造记忆后端
memory = AMemBackend(
    config=AMemConfig(),
    llm=llm,
)

# 传给 Agent
agent = agent.Agent(memory=memory, ...)
```

## AMemConfig 消融实验旋钮

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `recall_k` | 5 | 每次 retrieve 返回的最大条数 |
| `inject_field` | `"text"` | `"text"` 只注入原文；`"content+keywords+context"` 注入增强字段 |
| `write_user_message` | `True` | 是否把用户消息写入记忆库 |
| `write_assistant_message` | `True` | 是否把助手回复写入记忆库 |
| `write_tool_actions` | `False` | 是否把工具调用结果写入记忆库 |
| `enable_metadata_extraction` | `True` | 关闭后退化为 naive RAG（纯向量，无 LLM 提取） |
| `enable_evolution` | `True` | 关闭后跳过 3-call evolution |
| `evolution_threshold` | 3 | 新 note 至少有 N 个邻居才触发 evolution |
| `evolution_neighbors_k` | 5 | evolution 考察的邻居数量 |
| `persist_dir` | `None` | `None` 使用内存存储；设置路径则用 ChromaDB 持久化 |
| `collection_name` | `"amem"` | ChromaDB collection 名称 |
| `embedding_model` | `"all-MiniLM-L6-v2"` | sentence-transformers 模型名 |
| `llm_max_retries` | 2 | LLM 调用失败时的重试次数 |

## 消融实验组配置示例

### Baseline：纯向量 RAG（无 LLM）

```python
cfg = AMemConfig(
    enable_metadata_extraction=False,
    enable_evolution=False,
)
memory = AMemBackend(config=cfg)  # llm 可以不传
```

### 实验组 A：加元数据提取，不做 evolution

```python
cfg = AMemConfig(
    enable_metadata_extraction=True,
    enable_evolution=False,
    inject_field="content+keywords+context",
)
memory = AMemBackend(config=cfg, llm=llm)
```

### 实验组 B：完整 A-MEM（元数据 + evolution）

```python
cfg = AMemConfig(
    enable_metadata_extraction=True,
    enable_evolution=True,
    evolution_threshold=3,
    evolution_neighbors_k=5,
)
memory = AMemBackend(config=cfg, llm=llm)
```

### 实验组 C：持久化跨会话记忆

```python
cfg = AMemConfig(
    persist_dir="/path/to/chroma_db",
    collection_name="experiment_c",
)
memory = AMemBackend(config=cfg, llm=llm)
```

## 架构说明

```
AMemBackend (backend.py)
├── AMemConfig          — 所有旋钮
├── SentenceTransformerEmbedder (embedder.py)  — 向量化
├── InMemoryStore / ChromaStore (store.py)     — 向量存储
├── MetadataExtractor (extractor.py)           — LLM 提取 keywords/context/tags
└── EvolutionPolicy (evolution.py)             — 3-call evolution
    ├── Call 1: 决策 (NO_EVOLUTION / STRENGTHEN / UPDATE_NEIGHBOR / STRENGTHEN_AND_UPDATE)
    ├── Call 2: strengthen（更新 new_note 的 links/tags）
    └── Call 3: update_neighbors（改写邻居的 context/tags）
```

### 两层存储分离

- `_raw_messages`：原始对话流，`messages()` 返回给引擎用于构建 prompt
- `_store`（VectorStore）：经过元数据提取和 evolution 演化的 MemoryNote，`retrieve()` 从这里检索

### 脱离 AgentSocket 独立使用

A-MEM 内部不 import AgentSocket 的 clients 模块，可以用任意 LLMClient 实现独立测试：

```python
class MockLLM:
    def complete(self, messages, **kwargs) -> str:
        return "KEYWORDS: test\nCONTEXT: test.\nTAGS: test"

memory = AMemBackend(config=AMemConfig(), llm=MockLLM())
```
