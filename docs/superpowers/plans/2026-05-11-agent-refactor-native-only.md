# Agent Refactor: Native Tool Calling Only — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the ReAct text-parsing path, clean up adapters, add `agent.py` as the single entry point, rename clients to `AnthropicClient`/`OpenAIClient`, and wire the tracer in `NativeToolEngine`.

**Architecture:** Single execution engine (`NativeToolEngine`) using the model's native function calling API. LLM configuration is fully decoupled from agent configuration — `Agent` accepts a pre-built client. All public types are re-exported from the top-level `__init__.py`.

**Tech Stack:** Python 3.11+, `anthropic` SDK, `openai` SDK, `pytest`

---

### Task 1: Set up pytest

**Files:**
- Create: `pyproject.toml`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "agent"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `tests/__init__.py`**

```python
```

(empty file)

- [ ] **Step 3: Verify pytest is available**

Run: `python -m pytest --collect-only`
Expected: `no tests ran` (no errors)

- [ ] **Step 4: Commit**

```bash
git init
git add pyproject.toml tests/__init__.py
git commit -m "chore: add pytest setup"
```

---

### Task 2: Delete ReAct files and adapters directory

**Files:**
- Delete: `core/engine.py`
- Delete: `core/protocol.py`
- Delete: `core/parser.py`
- Delete: `modular_agent.py`
- Delete: `adapters/langchain.py`
- Delete: `adapters/langgraph.py`
- Delete: `adapters/__init__.py`

- [ ] **Step 1: Delete ReAct core files**

```bash
rm agent/core/engine.py agent/core/protocol.py agent/core/parser.py
```

- [ ] **Step 2: Delete modular_agent.py**

```bash
rm agent/modular_agent.py
```

- [ ] **Step 3: Delete adapters directory**

```bash
rm -r agent/adapters
```

- [ ] **Step 4: Verify no import errors in remaining files**

Run: `python -c "from agent.core import engine_native, experiment, memory, middleware, tools, types"`
Expected: no output (clean import)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove ReAct engine, parser, protocol, and adapters"
```

---

### Task 3: Slim down `clients.py` — remove ReAct-only methods, rename classes

**Files:**
- Modify: `agent/clients.py`

- [ ] **Step 1: Write a test that imports the new names and verifies the old ones are gone**

Create `tests/test_clients.py`:

```python
import pytest


def test_anthropic_client_importable():
    from agent.clients import AnthropicClient
    assert AnthropicClient is not None


def test_openai_client_importable():
    from agent.clients import OpenAIClient
    assert OpenAIClient is not None


def test_old_names_gone():
    import agent.clients as m
    assert not hasattr(m, "OpenAIModelClient")
    assert not hasattr(m, "AnthropicModelClient")


def test_generate_method_gone():
    from agent.clients import AnthropicClient
    assert not hasattr(AnthropicClient, "generate")
    assert not hasattr(AnthropicClient, "generate_with_blocks")


def test_call_with_tools_present():
    from agent.clients import AnthropicClient, OpenAIClient
    assert hasattr(AnthropicClient, "call_with_tools")
    assert hasattr(OpenAIClient, "call_with_tools")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_clients.py -v`
Expected: `test_old_names_gone` and `test_generate_method_gone` FAIL

- [ ] **Step 3: Rewrite `clients.py`**

Replace the full content of `agent/clients.py` with:

```python
"""Model client implementations for OpenAI-compatible and Anthropic APIs.

Environment variables
---------------------
Common:
  LLM_PROVIDER      "openai" | "anthropic"  (default: anthropic)
  AGENT_MODEL       model name              (default: provider-specific)

OpenAI-compatible (OpenAI, vLLM, Ollama, DeepSeek, ...):
  OPENAI_API_KEY    required
  OPENAI_BASE_URL   optional (default: https://api.openai.com/v1)
  AGENT_MODEL       default: gpt-4o

Anthropic:
  ANTHROPIC_API_KEY  required
  ANTHROPIC_BASE_URL optional (default: Anthropic SDK default)
  AGENT_MODEL        default: claude-opus-4-7
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export "):].lstrip()
    if "=" not in line:
        return None
    key, raw_value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = raw_value.strip()
    if (
        len(value) >= 2
        and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"))
    ):
        return key, value[1:-1]
    value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
    return key, value


def _load_dotenv_if_present(path: Path | None = None) -> None:
    dotenv_path = path or (Path.cwd() / ".env")
    if not dotenv_path.exists():
        return
    try:
        content = dotenv_path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in content.splitlines():
        parsed = _parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def _value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


class OpenAIClient:
    """Talks to any OpenAI-compatible chat API (OpenAI, vLLM, Ollama, DeepSeek, etc.)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **extra_kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is required: pip install openai") from exc

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        resolved_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        self.model = model or os.environ.get("AGENT_MODEL", "gpt-4o")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs
        self._client = OpenAI(api_key=resolved_key, base_url=resolved_url)

    def call_with_tools(
        self,
        messages: list[dict],
        tools: list,
    ) -> "NativeModelResponse":
        from agent.core.types import NativeModelResponse, NativeToolCall
        import json

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            tools=openai_tools,  # type: ignore[arg-type]
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            **self.extra_kwargs,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            native_calls = [
                NativeToolCall(
                    call_id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in msg.tool_calls
            ]
            raw_assistant = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
            return NativeModelResponse(
                final_answer=None,
                tool_calls=native_calls,
                raw_assistant_message=raw_assistant,
            )

        return NativeModelResponse(
            final_answer=msg.content or "",
            tool_calls=[],
            raw_assistant_message={"role": "assistant", "content": msg.content or ""},
        )

    def tool_result_message(self, call_id: str, _tool_name: str, result: str) -> dict:
        return {"role": "tool", "tool_call_id": call_id, "content": result}


class AnthropicClient:
    """Talks to the Anthropic Messages API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **extra_kwargs: Any,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError("anthropic package is required: pip install anthropic") from exc

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        resolved_url = base_url or os.environ.get("ANTHROPIC_BASE_URL") or None
        self.model = model or os.environ.get("AGENT_MODEL", "claude-opus-4-7")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs
        kwargs: dict[str, Any] = {"api_key": resolved_key}
        if resolved_url:
            kwargs["base_url"] = resolved_url
        self._client = _anthropic.Anthropic(**kwargs)

    def call_with_tools(
        self,
        messages: list[dict],
        tools: list,
        system: str = "",
    ) -> "NativeModelResponse":
        from agent.core.types import NativeModelResponse, NativeToolCall

        anthropic_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters_schema or {"type": "object", "properties": {}},
            }
            for t in tools
        ]

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
            "tools": anthropic_tools,
            **self.extra_kwargs,
        }
        if system:
            create_kwargs["system"] = system

        response = self._client.messages.create(**create_kwargs)

        tool_use_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        text_blocks = [b for b in response.content if getattr(b, "type", None) == "text"]

        if tool_use_blocks:
            native_calls = [
                NativeToolCall(
                    call_id=b.id,
                    name=b.name,
                    arguments=dict(b.input),
                )
                for b in tool_use_blocks
            ]
            raw_content = [
                {"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)}
                for b in tool_use_blocks
            ]
            if text_blocks:
                raw_content = [{"type": "text", "text": text_blocks[0].text}] + raw_content
            return NativeModelResponse(
                final_answer=None,
                tool_calls=native_calls,
                raw_assistant_message={"role": "assistant", "content": raw_content},
            )

        text = text_blocks[0].text if text_blocks else ""
        return NativeModelResponse(
            final_answer=text,
            tool_calls=[],
            raw_assistant_message={"role": "assistant", "content": text},
        )

    def tool_result_message(self, call_id: str, _tool_name: str, result: str) -> dict:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": result}],
        }


def from_env(
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> OpenAIClient | AnthropicClient:
    """Instantiate the right client based on LLM_PROVIDER env var."""
    _load_dotenv_if_present()
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        return OpenAIClient(max_tokens=max_tokens, temperature=temperature)
    if provider == "anthropic":
        return AnthropicClient(max_tokens=max_tokens, temperature=temperature)
    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Expected 'openai' or 'anthropic'."
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_clients.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent/clients.py tests/test_clients.py
git commit -m "refactor: rename clients to AnthropicClient/OpenAIClient, remove generate() path"
```

---

### Task 4: Wire tracer in `NativeToolEngine`

**Files:**
- Modify: `agent/core/engine_native.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_engine_native_tracer.py`:

```python
import time
from dataclasses import dataclass, field
from agent.core.engine_native import NativeToolEngine
from agent.core.memory import InMemorySessionMemory
from agent.core.middleware import MiddlewareChain
from agent.core.tools import ToolRegistry
from agent.core.types import (
    AgentRunResult, NativeModelResponse, NativeToolCall,
    RuntimeConfig, ToolCallTrace, ToolSpec,
)


@dataclass
class CapturingTracer:
    records: list[ToolCallTrace] = field(default_factory=list)

    def record(self, trace: ToolCallTrace) -> None:
        self.records.append(trace)


class FakeClient:
    """Returns one tool call then a final answer."""
    def __init__(self):
        self._call_count = 0

    def call_with_tools(self, messages, tools, system=""):
        self._call_count += 1
        if self._call_count == 1:
            return NativeModelResponse(
                final_answer=None,
                tool_calls=[NativeToolCall(call_id="c1", name="echo", arguments={"msg": "hi"})],
                raw_assistant_message={"role": "assistant", "content": []},
            )
        return NativeModelResponse(
            final_answer="done",
            tool_calls=[],
            raw_assistant_message={"role": "assistant", "content": "done"},
        )

    def tool_result_message(self, call_id, tool_name, result):
        return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": result}]}


def test_tracer_records_tool_call():
    tracer = CapturingTracer()
    tool = ToolSpec(
        name="echo",
        description="echo",
        handler=lambda args: args["msg"],
        parameters_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
    )
    registry = ToolRegistry(tools=[tool], tracer=tracer)
    engine = NativeToolEngine(
        model_client=FakeClient(),
        tool_registry=registry,
        memory=InMemorySessionMemory(),
        middleware_chain=MiddlewareChain([]),
        runtime_config=RuntimeConfig(),
    )
    result = engine.run("say hi")
    assert result.final_answer == "done"
    assert len(tracer.records) == 1
    trace = tracer.records[0]
    assert trace.tool_name == "echo"
    assert trace.arguments == {"msg": "hi"}
    assert trace.result_summary == "hi"
    assert trace.middleware_decision == "allow"
    assert trace.duration_ms >= 0
    assert trace.timestamp > 0


def test_tracer_none_no_error():
    """No tracer set — engine runs without error."""
    tool = ToolSpec(
        name="echo",
        description="echo",
        handler=lambda args: args["msg"],
        parameters_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
    )
    registry = ToolRegistry(tools=[tool])
    engine = NativeToolEngine(
        model_client=FakeClient(),
        tool_registry=registry,
        memory=InMemorySessionMemory(),
        middleware_chain=MiddlewareChain([]),
        runtime_config=RuntimeConfig(),
    )
    result = engine.run("say hi")
    assert result.final_answer == "done"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_engine_native_tracer.py -v`
Expected: `test_tracer_records_tool_call` FAIL (tracer never called)

- [ ] **Step 3: Add tracer call in `engine_native.py`**

In `NativeToolEngine.run()`, after `result = self.middleware_chain.after_tool_call(...)` and before `self.memory.append_action(...)`, add:

```python
tool_end = time.time()
if self.tool_registry.tracer is not None:
    self.tool_registry.tracer.record(
        ToolCallTrace(
            session_id=session_id,
            step_index=len(steps),
            tool_name=tc.name,
            arguments=action_input,
            result_summary=result.summary,
            duration_ms=(tool_end - tool_start) * 1000,
            middleware_decision=decision.action,
            timestamp=tool_start,
        )
    )
```

Also add `tool_start = time.time()` just before `raw_result = tool.handler(dict(action_input))`.

Import `ToolCallTrace` at the top of `engine_native.py` — update the existing import line:

```python
from agent.core.types import AgentRunResult, MemoryBackend, ParsedStep, RuntimeConfig, ToolCallRecord, ToolCallTrace, ToolResult
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_engine_native_tracer.py -v`
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent/core/engine_native.py tests/test_engine_native_tracer.py
git commit -m "feat: wire ToolCallTracer in NativeToolEngine"
```

---

### Task 5: Create `agent.py`

**Files:**
- Create: `agent/agent.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_agent.py`:

```python
from unittest.mock import MagicMock
from agent.agent import Agent
from agent.core.types import (
    NativeModelResponse, NativeToolCall, RuntimeConfig, ToolSpec,
)
from agent.core.memory import InMemorySessionMemory
from agent.core.middleware import MiddlewareChain


def _make_client(final_answer="done"):
    client = MagicMock()
    client.call_with_tools.return_value = NativeModelResponse(
        final_answer=final_answer,
        tool_calls=[],
        raw_assistant_message={"role": "assistant", "content": final_answer},
    )
    return client


def test_agent_run_returns_result():
    client = _make_client("hello")
    agent = Agent(model_client=client, tools=[])
    result = agent.run("hi")
    assert result.final_answer == "hello"
    assert result.stop_reason == "finished"


def test_agent_defaults_are_applied():
    """Agent constructs defaults when optional args are None."""
    client = _make_client()
    agent = Agent(model_client=client, tools=[])
    assert agent._engine is not None


def test_agent_custom_system_prompt():
    client = _make_client()
    agent = Agent(model_client=client, tools=[], system_prompt="You are a pirate.")
    result = agent.run("hello")
    call_kwargs = client.call_with_tools.call_args
    # system prompt is passed through to the engine
    assert result.final_answer == "done"


def test_agent_custom_runtime_config():
    client = _make_client()
    config = RuntimeConfig(max_steps=3)
    agent = Agent(model_client=client, tools=[], runtime_config=config)
    assert agent._engine.runtime_config.max_steps == 3


def test_agent_custom_memory():
    client = _make_client()
    memory = InMemorySessionMemory()
    agent = Agent(model_client=client, tools=[], memory=memory)
    agent.run("test")
    assert len(memory.messages()) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent.py -v`
Expected: ImportError — `agent.agent` does not exist

- [ ] **Step 3: Create `agent/agent.py`**

```python
from __future__ import annotations

from agent.core.engine_native import NativeToolEngine
from agent.core.memory import InMemorySessionMemory
from agent.core.middleware import MiddlewareChain
from agent.core.tools import ToolRegistry
from agent.core.types import AgentRunResult, MemoryBackend, RuntimeConfig, ToolCallTracer, ToolSpec


class Agent:
    """Main entry point for the agent framework.

    LLM configuration lives in the model_client (AnthropicClient / OpenAIClient).
    All other modules are optional and default to sensible no-op implementations.
    """

    def __init__(
        self,
        model_client: object,
        tools: list[ToolSpec],
        memory: MemoryBackend | None = None,
        middleware_chain: MiddlewareChain | None = None,
        runtime_config: RuntimeConfig | None = None,
        system_prompt: str | None = None,
        tracer: ToolCallTracer | None = None,
    ) -> None:
        resolved_memory = memory or InMemorySessionMemory()
        resolved_middleware = middleware_chain or MiddlewareChain([])
        resolved_config = runtime_config or RuntimeConfig()
        registry = ToolRegistry(tools=tools, tracer=tracer)

        engine_kwargs: dict = dict(
            model_client=model_client,
            tool_registry=registry,
            memory=resolved_memory,
            middleware_chain=resolved_middleware,
            runtime_config=resolved_config,
        )
        if system_prompt is not None:
            engine_kwargs["system_prompt"] = system_prompt

        self._engine = NativeToolEngine(**engine_kwargs)

    def run(self, user_input: str) -> AgentRunResult:
        return self._engine.run(user_input)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_agent.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent/agent.py tests/test_agent.py
git commit -m "feat: add Agent entry point"
```

---

### Task 6: Update `__init__.py`

**Files:**
- Modify: `agent/__init__.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_public_api.py`:

```python
def test_top_level_imports():
    from agent import (
        Agent,
        AnthropicClient,
        OpenAIClient,
        from_env,
        ToolSpec,
        RuntimeConfig,
        AgentRunResult,
        ToolCallTrace,
        MemoryBackend,
        ToolCallTracer,
        InMemorySessionMemory,
        MiddlewareChain,
        ToolRegistry,
    )
    assert Agent is not None
    assert AnthropicClient is not None
    assert OpenAIClient is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_public_api.py -v`
Expected: ImportError

- [ ] **Step 3: Update `agent/__init__.py`**

```python
"""Argus modular agent package."""

from agent.agent import Agent
from agent.clients import AnthropicClient, OpenAIClient, from_env
from agent.core.memory import InMemorySessionMemory
from agent.core.middleware import MiddlewareChain
from agent.core.tools import ToolRegistry
from agent.core.types import (
    AgentRunResult,
    MemoryBackend,
    RuntimeConfig,
    ToolCallTrace,
    ToolCallTracer,
    ToolSpec,
)

__all__ = [
    "Agent",
    "AnthropicClient",
    "OpenAIClient",
    "from_env",
    "InMemorySessionMemory",
    "MiddlewareChain",
    "ToolRegistry",
    "AgentRunResult",
    "MemoryBackend",
    "RuntimeConfig",
    "ToolCallTrace",
    "ToolCallTracer",
    "ToolSpec",
]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_public_api.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add agent/__init__.py tests/test_public_api.py
git commit -m "feat: expose public API from top-level __init__.py"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest -v`
Expected: all tests PASS, no warnings about missing modules

- [ ] **Step 2: Verify deleted files are gone**

```bash
python -c "
import os
deleted = [
    'agent/core/engine.py',
    'agent/core/protocol.py',
    'agent/core/parser.py',
    'agent/modular_agent.py',
    'agent/adapters',
]
for path in deleted:
    assert not os.path.exists(path), f'Should be deleted: {path}'
print('All deleted files confirmed gone.')
"
```

Expected: `All deleted files confirmed gone.`

- [ ] **Step 3: Verify clean import of full public API**

```bash
python -c "
from agent import Agent, AnthropicClient, OpenAIClient, from_env
from agent import ToolSpec, RuntimeConfig, AgentRunResult
from agent import InMemorySessionMemory, MiddlewareChain, ToolRegistry
print('All public imports OK.')
"
```

Expected: `All public imports OK.`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final verification — native-only refactor complete"
```
