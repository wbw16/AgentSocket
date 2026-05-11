from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from agent.core.types import AgentRunResult, MemoryBackend


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class JsonlRunLogger:
    """Append agent run records as JSON Lines."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record_run(
        self,
        *,
        user_input: str,
        result: AgentRunResult,
        memory: MemoryBackend | None = None,
        experiment_name: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "run_id": result.session_id,
            "timestamp": time.time(),
            "experiment_name": experiment_name,
            "input": user_input,
            "final_answer": result.final_answer,
            "stop_reason": result.stop_reason,
            "metrics": _jsonable(dict(result.metrics)),
            "action_history": _jsonable(result.action_history),
            "memory_messages": _jsonable(memory.messages()) if memory is not None else [],
        }

        snapshot = getattr(memory, "snapshot", None) if memory is not None else None
        if callable(snapshot):
            record["memory_snapshot"] = _jsonable(snapshot())

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
