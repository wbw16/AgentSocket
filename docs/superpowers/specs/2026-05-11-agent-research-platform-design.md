# Agent Research Platform Design

**Date:** 2026-05-11  
**Scope:** Tool-use research + agent memory research (episodic, semantic, working memory)  
**Goal:** Extend the existing modular agent framework into a general-purpose research base that supports pluggable memory backends and structured experiment comparison.

---

## Context

The existing codebase has a clean dual-engine architecture (ReAct text-parsing + native tool calling), multi-provider support (OpenAI-compatible + Anthropic), and a middleware pipeline. The current memory implementation (`InMemorySessionMemory`) is a single concrete class wired directly into both engines.

The changes needed are minimal and targeted — no architectural overhaul.

---

## Changes

### 1. `MemoryBackend` Protocol (`core/types.py`)

Define a `MemoryBackend` Protocol that all memory implementations must satisfy:

```python
class MemoryBackend(Protocol):
    def append_message(self, role: str, content: str) -> None: ...
    def append_action(self, record: ToolCallRecord) -> None: ...
    def messages(self) -> list[dict[str, str]]: ...
    def action_history(self) -> list[ToolCallRecord]: ...
    def retrieve(self, query: str, k: int = 5) -> list[dict]: ...
    def summarize(self, max_tokens: int) -> str: ...
```

`retrieve` and `summarize` are extension points for semantic and working memory research. The default `InMemorySessionMemory` implements them as no-ops (returns `[]` and `""` respectively).

Future implementations (SQLite, vector store, knowledge graph, etc.) are out of scope for this design — the Protocol is the only contract.

### 2. `InMemorySessionMemory` implements `MemoryBackend` (`core/memory.py`)

Add `retrieve` and `summarize` stubs so the existing class satisfies the Protocol. No behavioral change.

### 3. `ToolCallTrace` dataclass (`core/types.py`)

```python
@dataclass(slots=True)
class ToolCallTrace:
    session_id: str
    step_index: int
    tool_name: str
    arguments: dict
    result_summary: str
    duration_ms: float
    middleware_decision: str  # "allow" | "deny" | "rewrite" | "escalate"
    timestamp: float
```

### 4. `ToolCallTracer` Protocol + `ToolRegistry` update (`core/tools.py`)

```python
class ToolCallTracer(Protocol):
    def record(self, trace: ToolCallTrace) -> None: ...
```

`ToolRegistry` gains an optional `tracer: ToolCallTracer | None = None`. When set, it records a `ToolCallTrace` after every tool execution. When `None` (default), zero overhead.

### 5. Engine type updates (`core/engine.py`, `core/engine_native.py`)

Change the `memory` field type annotation from `InMemorySessionMemory` to `MemoryBackend` in both engines. No logic changes.

### 6. `ExperimentHarness` (`core/experiment.py`) — new file

```python
@dataclass
class ExperimentConfig:
    name: str
    memory_factory: Callable[[], MemoryBackend]
    tools: list[ToolSpec]
    model_client: ModelClient
    inputs: list[str]
    runtime_config: RuntimeConfig = field(default_factory=RuntimeConfig)

class ExperimentHarness:
    def run(self, configs: list[ExperimentConfig]) -> dict[str, list[AgentRunResult]]:
        ...
```

Returns `{config_name: [AgentRunResult, ...]}` — raw results only. Metric computation is left to the researcher (pandas, notebook, wandb, etc.).

---

## What this does NOT include

- Concrete memory implementations (SQLite, vector store, knowledge graph) — added by researcher as needed
- Metric aggregation or visualization — out of scope, use external tools
- Changes to the middleware system, protocol, or parser

---

## Backward compatibility

All changes are additive or type-annotation-only. Existing code using `InMemorySessionMemory` directly continues to work unchanged. Not passing a `tracer` to `ToolRegistry` is zero-cost.
