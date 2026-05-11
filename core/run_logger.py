from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from agent.core.types import AgentRunResult, MemoryBackend


def _jsonable(value: Any) -> Any:
    """内部辅助函数，将传入对象转换成适用于 json 序列化保存的字典或列表结构。"""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class JsonlRunLogger:
    """以 JSON Lines (jsonl) 格式，将模型单次运行的记录不断追加到文本当中。"""

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
        """记录一组 Agent 的运行数据并存储到日志文件中。"""
        # 构建运行记录结构体
        record: dict[str, Any] = {
            "run_id": result.session_id,                      # 运行 ID
            "timestamp": time.time(),                         # 发生时间的时间戳
            "experiment_name": experiment_name,               # 所属实验名称(可选)
            "input": user_input,                              # 用户原始请求
            "final_answer": result.final_answer,              # Agent的最终回答
            "stop_reason": result.stop_reason,                # 为什么结束运行的原因
            "metrics": _jsonable(dict(result.metrics)),       # 各项性能参数指标
            "action_history": _jsonable(result.action_history), # 使用工具的历史记录
            "memory_messages": _jsonable(memory.messages()) if memory is not None else [], # 发给模型的完整历史对话
        }

        # 尝试将整个记忆的结构或者快照加入进去进行存档
        snapshot = getattr(memory, "snapshot", None) if memory is not None else None
        if callable(snapshot):
            record["memory_snapshot"] = _jsonable(snapshot())

        # 确保目录存在
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 用 utf-8 编码以及非 ascii 编码的 json 追加日志
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
