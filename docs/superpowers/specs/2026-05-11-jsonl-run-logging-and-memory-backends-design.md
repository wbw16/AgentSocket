# JSONL Run Logging and Memory Backends Design

## Goal

Support memory-mechanism research by making agent runs easy to persist and compare in bulk.

The implementation should stay deliberately small:

- Persist each agent run or experiment item as one JSONL record.
- Keep the memory interface stable and minimal.
- Provide a `memory_backends/` folder where each tested memory mechanism can live in its own file.
- Include one tiny memory backend demo for smoke testing.

## Non-Goals

- No SQL export.
- No registry, plugin system, or memory taxonomy.
- No predefined research categories such as summary/vector/episodic/hybrid memory.
- No opinionated experiment database.

## File Layout

```text
agent/
├── core/
│   ├── memory.py          # MemoryBackend Protocol + InMemorySessionMemory
│   ├── run_logger.py      # JsonlRunLogger
│   └── experiment.py      # ExperimentHarness can optionally write JSONL
├── memory_backends/
│   ├── __init__.py
│   └── simple_demo.py     # Minimal demo memory for smoke tests
└── tests/
    ├── test_run_logger.py
    ├── test_experiment_logging.py
    └── test_memory_backends.py
```

## Memory Backend Rule

`core/memory.py` remains the stable contract. A memory backend is any class that implements:

```python
append_message(role: str, content: str) -> None
append_action(record: ToolCallRecord) -> None
messages() -> list[dict[str, str]]
action_history() -> list[ToolCallRecord]
retrieve(query: str, k: int = 5) -> list[dict]
summarize(max_tokens: int) -> str
```

Research memory implementations should be placed in `memory_backends/`, usually one file per mechanism:

```text
memory_backends/my_memory_v1.py
memory_backends/my_memory_v2.py
memory_backends/ablation_no_retrieval.py
```

They do not need registration. Experiments pass them directly through `memory_factory`.

## Simple Demo Memory

Create `memory_backends/simple_demo.py` with `SimpleDemoMemory`.

Its purpose is smoke testing, not research quality:

- Store messages and tool actions in lists.
- Return stored messages from `messages()`.
- Return stored actions from `action_history()`.
- Return the last `k` messages from `retrieve()`.
- Return a short joined text from `summarize(max_tokens)`.
- Optionally expose `snapshot()` so the JSONL logger can persist internal state.

Example use:

```python
from agent.memory_backends.simple_demo import SimpleDemoMemory

agent = Agent(
    model_client=llm,
    tools=[weather_tool],
    memory=SimpleDemoMemory(),
)
```

Experiment use:

```python
ExperimentConfig(
    name="simple-demo",
    memory_factory=SimpleDemoMemory,
    tools=[weather_tool],
    model_client=llm,
    inputs=inputs,
)
```

## JSONL Run Logger

Create `core/run_logger.py` with `JsonlRunLogger`.

The logger appends one UTF-8 JSON object per line. It should create parent directories automatically.

Record shape:

```json
{
  "run_id": "uuid",
  "timestamp": 1710000000.0,
  "experiment_name": "simple-demo",
  "input": "user input",
  "final_answer": "assistant output",
  "stop_reason": "finished",
  "metrics": {},
  "action_history": [],
  "memory_messages": [],
  "memory_snapshot": {}
}
```

Field rules:

- `run_id` should come from `AgentRunResult.session_id`.
- `timestamp` is logger write time.
- `experiment_name` is optional and may be `None` for direct `Agent` use.
- `action_history` converts `ToolCallRecord` dataclasses to JSON-friendly dicts.
- `memory_messages` comes from `memory.messages()` when a memory object is available.
- `memory_snapshot` is included only when memory has a callable `snapshot()` method.

## Integration Points

### Direct Agent Runs

`Agent` can optionally accept a `run_logger`.

When present, `Agent.run(user_input)` should:

1. Call the engine.
2. Log the input, result, and memory snapshot.
3. Return the unchanged `AgentRunResult`.

### Experiment Harness

`ExperimentHarness.run()` can optionally accept a logger or each `ExperimentConfig` can carry one logging path. Prefer the smallest API:

```python
harness = ExperimentHarness(run_logger=JsonlRunLogger("runs/experiment.jsonl"))
```

For each input, the harness logs:

- `experiment_name=config.name`
- `input=user_input`
- `result=AgentRunResult`
- `memory=memory`

## Testing

Use TDD.

Tests should cover:

- `JsonlRunLogger` creates the JSONL file and appends one valid JSON object per run.
- The logger serializes `AgentRunResult`, action history, memory messages, and optional `snapshot()`.
- `Agent(..., run_logger=logger).run(...)` logs one line while preserving the returned result.
- `ExperimentHarness(run_logger=logger)` logs one line per input/config pair.
- `SimpleDemoMemory` satisfies the memory interface and has an inspectable snapshot.

## README Update

Update README with:

- JSONL logging example.
- `memory_backends/` convention.
- `SimpleDemoMemory` smoke-test example.
