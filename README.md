# Agent

一个轻量、模块化的 Python Agent 框架，只使用大模型原生 Function Calling / Tool Use。当前版本已经移除 ReAct 文本解析路径：模型调用、工具调用、工具结果回传都走结构化 API。

核心设计：

- `Agent` 是唯一主入口。
- LLM 配置完全放在 `AnthropicClient` / `OpenAIClient` 中，`Agent` 只接收预先构建好的 client。
- 工具通过 `ToolSpec` 注册，并用 JSON Schema 暴露给模型。
- 记忆、中间件、运行配置、工具追踪都可以替换。
- 常用类型从顶层 `agent` 包导出。

---

## 目录

- [安装](#安装)
- [快速上手](#快速上手)
- [LLM 客户端](#llm-客户端)
- [工具注册](#工具注册)
- [Agent 完整配置](#agent-完整配置)
- [中间件](#中间件)
- [自定义记忆后端](#自定义记忆后端)
- [记忆机制研究目录](#记忆机制研究目录)
- [JSONL 运行日志](#jsonl-运行日志)
- [工具调用追踪](#工具调用追踪)
- [实验对比](#实验对比)
- [开发与测试](#开发与测试)
- [项目结构](#项目结构)

---

## 安装

要求 Python 3.11+。

```bash
pip install anthropic          # 使用 Anthropic 模型
pip install openai             # 使用 OpenAI 或兼容接口
```

克隆后以可编辑模式安装：

```bash
pip install -e .
```

如果只是本地运行测试：

```bash
python -m pytest -v
```

---

## 快速上手

```python
from agent import Agent, AnthropicClient, ToolSpec

# 1. 定义工具
def get_weather(args: dict) -> str:
    city = args["city"]
    return f"{city} 今天晴，25°C"

weather_tool = ToolSpec(
    name="get_weather",
    description="查询指定城市的天气",
    handler=get_weather,
    parameters_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"}
        },
        "required": ["city"],
    },
)

# 2. 创建 LLM 客户端
llm = AnthropicClient(
    api_key="your-api-key",
    model="claude-opus-4-7",
)

# 3. 创建 Agent 并运行
agent = Agent(model_client=llm, tools=[weather_tool])
result = agent.run("北京今天天气怎么样？")
print(result.final_answer)
```

---

## LLM 客户端

LLM 配置完全独立于 Agent，先构建客户端再传入 Agent。

### Anthropic

```python
from agent import AnthropicClient

llm = AnthropicClient(
    api_key="your-api-key",          # 不传则读 ANTHROPIC_API_KEY 环境变量
    base_url="https://...",          # 可选，自定义端点
    model="claude-opus-4-7",         # 不传则读 AGENT_MODEL 环境变量
    max_tokens=4096,
    temperature=0.0,
)
```

### OpenAI 及兼容接口

```python
from agent import OpenAIClient

# OpenAI
llm = OpenAIClient(api_key="your-api-key", model="gpt-4o")

# DeepSeek
llm = OpenAIClient(
    api_key="your-api-key",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
)

# 本地 Ollama
llm = OpenAIClient(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="llama3",
)
```

### 从环境变量自动选择

```bash
export LLM_PROVIDER=anthropic       # 或 openai
export ANTHROPIC_API_KEY=sk-...
export AGENT_MODEL=claude-opus-4-7
```

```python
from agent import from_env

llm = from_env()
```

`from_env()` 会读取当前工作目录下的 `.env`，但不会覆盖已经存在的环境变量。

---

## 工具注册

每个工具是一个 `ToolSpec`，包含名称、描述、处理函数和参数 Schema（JSON Schema 格式）。工具处理函数接收一个 `dict`，返回值会被转换为字符串摘要后传回模型。

```python
from agent import ToolSpec

def search_web(args: dict) -> str:
    query = args["query"]
    # 实际搜索逻辑...
    return f"搜索结果：{query}"

search_tool = ToolSpec(
    name="search_web",
    description="在网络上搜索信息",
    handler=search_web,
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"}
        },
        "required": ["query"],
    },
)
```

传入多个工具：

```python
agent = Agent(model_client=llm, tools=[weather_tool, search_tool])
```

---

## Agent 完整配置

```python
from agent import Agent, AnthropicClient, InMemorySessionMemory, MiddlewareChain, RuntimeConfig

agent = Agent(
    model_client=llm,                          # 必填
    tools=[weather_tool, search_tool],         # 必填
    memory=InMemorySessionMemory(),            # 可选，默认 InMemorySessionMemory
    middleware_chain=MiddlewareChain([]),       # 可选，默认空链
    runtime_config=RuntimeConfig(max_steps=8), # 可选
    system_prompt="你是一个专业的助手。",        # 可选
    tracer=None,                               # 可选，见「工具调用追踪」
)

result = agent.run("帮我查一下上海的天气")
print(result.final_answer)
print(result.stop_reason)      # "finished" | "max_steps_exceeded" | "denied" | ...
print(result.metrics)          # {"duration_seconds": 1.23}
```

### RuntimeConfig 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_steps` | `8` | 最大工具调用轮次 |
| `max_correction_retries` | `2` | 意图校正最大重试次数 |
| `stop_on_tool_error` | `True` | 工具报错时是否停止 |

---

## 中间件

中间件在工具调用前后执行，可用于权限控制、参数改写、结果过滤等。

```python
from agent import MiddlewareChain
from agent.core.types import MiddlewareDecision, ParsedStep, ToolResult, ToolSpec


class LoggingMiddleware:
    def before_tool_call(
        self, tool: ToolSpec, step: ParsedStep, action_input: dict
    ) -> MiddlewareDecision:
        print(f"[调用前] 工具: {tool.name}, 参数: {action_input}")
        return MiddlewareDecision(action="allow")

    def after_tool_call(
        self, tool: ToolSpec, step: ParsedStep, result: ToolResult
    ) -> ToolResult:
        print(f"[调用后] 结果: {result.summary}")
        return result


class DenyDangerousMiddleware:
    """拒绝调用高风险工具。"""
    def before_tool_call(self, tool, step, action_input) -> MiddlewareDecision:
        if tool.risk_level == "high":
            return MiddlewareDecision(action="deny", reason="高风险工具已被禁用")
        return MiddlewareDecision(action="allow")

    def after_tool_call(self, tool, step, result):
        return result


agent = Agent(
    model_client=llm,
    tools=[weather_tool],
    middleware_chain=MiddlewareChain([LoggingMiddleware(), DenyDangerousMiddleware()]),
)
```

`MiddlewareDecision.action` 可选值：

| 值 | 说明 |
|----|------|
| `"allow"` | 放行 |
| `"deny"` | 拒绝，Agent 停止并返回 `stop_reason="denied"` |
| `"rewrite"` | 改写参数，继续执行（需设置 `rewritten_input`） |
| `"escalate"` | 上报，Agent 停止并返回 `stop_reason="escalated"` |

---

## 自定义记忆后端

实现 `MemoryBackend` Protocol 即可替换默认的内存记忆：

```python
from agent.core.types import MemoryBackend, ToolCallRecord


class RedisMemory:
    """示例：基于 Redis 的持久化记忆。"""

    def __init__(self, redis_client, session_id: str):
        self._redis = redis_client
        self._session_id = session_id

    def append_message(self, role: str, content: str) -> None:
        self._redis.rpush(f"{self._session_id}:messages", f"{role}:{content}")

    def append_action(self, record: ToolCallRecord) -> None:
        self._redis.rpush(f"{self._session_id}:actions", str(record))

    def messages(self) -> list[dict[str, str]]:
        raw = self._redis.lrange(f"{self._session_id}:messages", 0, -1)
        result = []
        for item in raw:
            role, content = item.decode().split(":", 1)
            result.append({"role": role, "content": content})
        return result

    def action_history(self) -> list[ToolCallRecord]:
        return []

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        # 语义检索扩展点，默认返回空
        return []

    def summarize(self, max_tokens: int) -> str:
        # 记忆摘要扩展点，默认返回空
        return ""


agent = Agent(
    model_client=llm,
    tools=[weather_tool],
    memory=RedisMemory(redis_client, session_id="user-123"),
)
```

`MemoryBackend` 完整接口：

```python
class MemoryBackend(Protocol):
    def append_message(self, role: str, content: str) -> None: ...
    def append_action(self, record: ToolCallRecord) -> None: ...
    def messages(self) -> list[dict[str, str]]: ...
    def action_history(self) -> list[ToolCallRecord]: ...
    def retrieve(self, query: str, k: int = 5) -> list[dict]: ...   # 语义检索扩展点
    def summarize(self, max_tokens: int) -> str: ...                 # 摘要扩展点
```

---

## 记忆机制研究目录

后续测试多种记忆机制时，把每个机制放在 `memory_backends/` 下，通常一个机制一个 `.py` 文件。核心框架不会要求注册或插件配置，只要实现 `MemoryBackend` 接口即可。

```text
memory_backends/
├── __init__.py
├── simple_demo.py       # 最小 demo，方便冒烟测试
├── my_memory_v1.py      # 你的第一个实验记忆机制
├── my_memory_v2.py      # 你的第二个实验记忆机制
└── ...
```

内置的 `SimpleDemoMemory` 只用于确认链路通了：

```python
from agent import Agent, AnthropicClient, ToolSpec
from agent.memory_backends.simple_demo import SimpleDemoMemory

llm = AnthropicClient()

agent = Agent(
    model_client=llm,
    tools=[],
    memory=SimpleDemoMemory(),
)
```

批量实验时直接换 `memory_factory`：

```python
from agent.core.experiment import ExperimentConfig
from agent.memory_backends.simple_demo import SimpleDemoMemory

config = ExperimentConfig(
    name="simple-demo",
    memory_factory=SimpleDemoMemory,
    tools=[],
    model_client=llm,
    inputs=["测试输入 1", "测试输入 2"],
)
```

如果某个记忆机制实现了 `snapshot()`，JSONL logger 会自动把它的内部状态保存到 `memory_snapshot` 字段；没实现就跳过。

---

## JSONL 运行日志

`JsonlRunLogger` 会把每次 run 追加为一行 JSON，适合大量批量实验：

```python
from agent import Agent, JsonlRunLogger

agent = Agent(
    model_client=llm,
    tools=[weather_tool],
    memory=SimpleDemoMemory(),
    run_logger=JsonlRunLogger("runs/agent_runs.jsonl"),
)

result = agent.run("北京今天天气怎么样？")
```

每行记录包含：

```json
{
  "run_id": "...",
  "timestamp": 1710000000.0,
  "experiment_name": null,
  "input": "北京今天天气怎么样？",
  "final_answer": "...",
  "stop_reason": "finished",
  "metrics": {},
  "action_history": [],
  "memory_messages": [],
  "memory_snapshot": {}
}
```

批量实验也可以统一写入同一个 JSONL：

```python
from agent import JsonlRunLogger
from agent.core.experiment import ExperimentHarness

harness = ExperimentHarness(
    run_logger=JsonlRunLogger("runs/experiment.jsonl"),
)

results = harness.run([config])
```

---

## 工具调用追踪

实现 `ToolCallTracer` Protocol，每次工具执行后自动记录 `ToolCallTrace`：

```python
from agent import ToolCallTrace, ToolCallTracer


class FileTracer:
    def __init__(self, path: str):
        self._path = path

    def record(self, trace: ToolCallTrace) -> None:
        with open(self._path, "a") as f:
            f.write(
                f"{trace.timestamp:.3f}\t{trace.tool_name}\t"
                f"{trace.duration_ms:.1f}ms\t{trace.middleware_decision}\t"
                f"{trace.result_summary}\n"
            )


agent = Agent(
    model_client=llm,
    tools=[weather_tool],
    tracer=FileTracer("traces.tsv"),
)
```

`ToolCallTrace` 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | `str` | 本次 run 的唯一 ID |
| `step_index` | `int` | 当前步骤序号 |
| `tool_name` | `str` | 工具名称 |
| `arguments` | `dict` | 实际传入参数 |
| `result_summary` | `str` | 工具返回值的字符串表示 |
| `duration_ms` | `float` | 工具执行耗时（毫秒） |
| `middleware_decision` | `str` | 中间件决策：`allow`/`deny`/`rewrite`/`escalate` |
| `timestamp` | `float` | Unix 时间戳 |

---

## 实验对比

`ExperimentHarness` 用于对比不同记忆后端、工具集或模型配置的效果：

```python
from agent.core.experiment import ExperimentConfig, ExperimentHarness
from agent import InMemorySessionMemory, AnthropicClient, OpenAIClient, ToolSpec, RuntimeConfig

llm_claude = AnthropicClient(model="claude-opus-4-7")
llm_gpt = OpenAIClient(model="gpt-4o")

inputs = [
    "北京今天天气怎么样？",
    "上海明天会下雨吗？",
]

configs = [
    ExperimentConfig(
        name="claude-opus",
        model_client=llm_claude,
        memory_factory=InMemorySessionMemory,
        tools=[weather_tool],
        inputs=inputs,
        runtime_config=RuntimeConfig(max_steps=5),
    ),
    ExperimentConfig(
        name="gpt-4o",
        model_client=llm_gpt,
        memory_factory=InMemorySessionMemory,
        tools=[weather_tool],
        inputs=inputs,
        runtime_config=RuntimeConfig(max_steps=5),
    ),
]

harness = ExperimentHarness()
results = harness.run(configs)

# results = {"claude-opus": [AgentRunResult, ...], "gpt-4o": [AgentRunResult, ...]}
for name, runs in results.items():
    for i, run in enumerate(runs):
        print(f"[{name}] 输入 {i+1}: {run.final_answer} ({run.stop_reason})")
```

返回值是原始 `AgentRunResult` 列表，指标计算交给外部工具（pandas、wandb 等）。

---

## 开发与测试

运行全部测试：

```bash
python -m pytest -v
```

当前测试覆盖：

- `AnthropicClient` / `OpenAIClient` 新类名与 native tool calling 方法。
- 旧的 `OpenAIModelClient` / `AnthropicModelClient` 与 `generate()` 路径不再暴露。
- `NativeToolEngine` 对 `ToolCallTracer` 的调用。
- `Agent` 默认配置与自定义 memory/runtime config。
- `JsonlRunLogger` 的 JSONL 追加写入。
- `ExperimentHarness` 批量运行时的 JSONL 记录。
- `SimpleDemoMemory` 冒烟测试。
- 顶层 public API 导出。

---

## 项目结构

```
agent/
├── __init__.py          # 公共 API 导出
├── agent.py             # Agent 主入口
├── clients.py           # AnthropicClient / OpenAIClient / from_env
├── memory_backends/
│   ├── __init__.py
│   └── simple_demo.py    # 最小记忆机制 demo
└── core/
    ├── engine_native.py # NativeToolEngine（执行引擎）
    ├── experiment.py    # ExperimentHarness
    ├── memory.py        # InMemorySessionMemory
    ├── middleware.py    # MiddlewareChain
    ├── run_logger.py    # JsonlRunLogger
    ├── tools.py         # ToolRegistry
    └── types.py         # 所有 dataclass / Protocol 定义
```

已移除的旧路径：

```text
core/engine.py
core/parser.py
core/protocol.py
modular_agent.py
adapters/
```
