# Agent Refactor: Native Tool Calling Only

**Date:** 2026-05-11
**Scope:** Remove ReAct text-parsing path, clean up adapters, add `agent.py` entry point, rename clients, wire tracer

---

## Goal

Simplify the framework to a single execution path (native function calling), expose a clean `Agent` entry point with fully injectable modules, and separate LLM configuration from agent configuration — consistent with LangChain's design philosophy.

---

## Changes

### 1. Delete ReAct path and adapters

Remove the following files entirely:

- `core/engine.py`
- `core/protocol.py`
- `core/parser.py`
- `modular_agent.py`
- `adapters/` (entire directory: `langchain.py`, `langgraph.py`, `__init__.py`)

### 2. Slim down `clients.py`

Remove all ReAct-only code:

- Delete `generate()`, `generate_with_blocks()`, `_generate_streaming_reasoning_blocks()`
- Delete `_normalize_for_openai()`, `_normalize_for_anthropic()`
- Keep `_load_dotenv_if_present()` and `_parse_dotenv_line()` as private helpers — still used by `from_env()`

Rename classes:

- `OpenAIModelClient` → `OpenAIClient`
- `AnthropicModelClient` → `AnthropicClient`

Public interface per client (unchanged behavior):

```python
class AnthropicClient:
    def __init__(self, *, api_key=None, base_url=None, model=None,
                 max_tokens=4096, temperature=0.0, **extra_kwargs): ...
    def call_with_tools(self, messages, tools, system=None) -> NativeModelResponse: ...
    def tool_result_message(self, call_id, tool_name, result) -> dict: ...

class OpenAIClient:
    def __init__(self, *, api_key=None, base_url=None, model=None,
                 max_tokens=4096, temperature=0.0, **extra_kwargs): ...
    def call_with_tools(self, messages, tools) -> NativeModelResponse: ...
    def tool_result_message(self, call_id, tool_name, result) -> dict: ...

def from_env(*, max_tokens=4096, temperature=0.0) -> AnthropicClient | OpenAIClient: ...
```

### 3. New `agent.py`

Single entry point. Constructs a `NativeToolEngine` internally and delegates `run()` to it.

```python
class Agent:
    def __init__(
        self,
        model_client,                        # AnthropicClient | OpenAIClient, required
        tools: list[ToolSpec],               # required
        memory: MemoryBackend | None = None,
        middleware_chain: MiddlewareChain | None = None,
        runtime_config: RuntimeConfig | None = None,
        system_prompt: str | None = None,
        tracer: ToolCallTracer | None = None,
    ): ...

    def run(self, user_input: str) -> AgentRunResult: ...
```

Defaults when `None`:
- `memory` → `InMemorySessionMemory()`
- `middleware_chain` → `MiddlewareChain([])`
- `runtime_config` → `RuntimeConfig()`
- `system_prompt` → `NativeToolEngine` default string
- `tracer` → `None` (zero overhead)

Usage:

```python
from agent import Agent
from agent.clients import AnthropicClient

llm = AnthropicClient(model="claude-opus-4-7", api_key="...")
agent = Agent(model_client=llm, tools=[...])
result = agent.run("帮我查一下天气")
```

### 4. Wire `tracer` in `NativeToolEngine`

After each tool execution, record a `ToolCallTrace` if `self.tool_registry.tracer` is set:

```python
if self.tool_registry.tracer is not None:
    self.tool_registry.tracer.record(ToolCallTrace(
        session_id=session_id,
        step_index=len(steps),
        tool_name=tc.name,
        arguments=action_input,
        result_summary=result.summary,
        duration_ms=(time.time() - tool_start) * 1000,
        middleware_decision=decision.action,
        timestamp=tool_start,
    ))
```

No behavior change when `tracer` is `None`.

### 5. Update `__init__.py`

Export the full public API so users only need `from agent import ...`:

```python
from agent.agent import Agent
from agent.clients import AnthropicClient, OpenAIClient, from_env
from agent.core.types import (
    ToolSpec, RuntimeConfig, AgentRunResult,
    ToolCallTrace, MemoryBackend, ToolCallTracer,
)
from agent.core.memory import InMemorySessionMemory
from agent.core.middleware import MiddlewareChain
from agent.core.tools import ToolRegistry
```

---

## Final structure

```
agent/
├── __init__.py
├── agent.py
├── clients.py
└── core/
    ├── __init__.py
    ├── engine_native.py
    ├── experiment.py
    ├── memory.py
    ├── middleware.py
    ├── tools.py
    └── types.py
```

---

## What this does NOT include

- New memory backends (SQLite, vector store) — researcher adds as needed
- LangGraph integration — out of scope; a real integration needs proper StateGraph design
- Metric aggregation or visualization — use external tools (pandas, wandb, etc.)

---

## Backward compatibility

Breaking changes (intentional):
- `OpenAIModelClient` / `AnthropicModelClient` renamed
- `generate()` / `generate_with_blocks()` removed
- `modular_agent.py` / `adapters/` removed

All other public interfaces (`NativeToolEngine`, `ExperimentHarness`, `ToolRegistry`, `MiddlewareChain`, `InMemorySessionMemory`) are unchanged.
