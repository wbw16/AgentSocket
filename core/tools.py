from __future__ import annotations

import time
from dataclasses import dataclass, field

from agent.core.types import ToolCallTracer, ToolSpec


@dataclass(slots=True)
class ToolRegistry:
    tools: list[ToolSpec]
    tracer: ToolCallTracer | None = None
    _by_name: dict[str, ToolSpec] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_name", {tool.name: tool for tool in self.tools})

    def get(self, name: str) -> ToolSpec:
        if name not in self._by_name:
            raise KeyError(f"Unknown tool: {name}")
        return self._by_name[name]

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools
        ]
