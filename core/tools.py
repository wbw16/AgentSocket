from __future__ import annotations

import time
from dataclasses import dataclass, field

from agent.core.types import ToolCallTracer, ToolSpec


@dataclass(slots=True)
class ToolRegistry:
    """
    工具注册表，用于管理和查找可用的工具。
    包含工具列表以及一个可选的调用追踪器(tracer)。
    """
    tools: list[ToolSpec]  # 工具规范列表
    tracer: ToolCallTracer | None = None  # 用于记录或追踪工具调用的追踪器，默认为空
    _by_name: dict[str, ToolSpec] = field(init=False, default_factory=dict)  # 按名称索引工具的内部字典

    def __post_init__(self) -> None:
        """
        数据类初始化后的回调函数。
        根据传入的tools列表构建按工具名称查找的字典 (_by_name)。
        """
        object.__setattr__(self, "_by_name", {tool.name: tool for tool in self.tools})

    def get(self, name: str) -> ToolSpec:
        """
        根据工具名称获取对应的工具规范 (ToolSpec)。
        如果在注册表中未找到，则抛出 KeyError。
        """
        if name not in self._by_name:
            raise KeyError(f"Unknown tool: {name}")
        return self._by_name[name]

    def describe(self) -> list[dict[str, str]]:
        """
        返回所有已注册工具的描述列表。
        每个描述由工具名称 (name) 和描述文本 (description) 组成。
        """
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools
        ]
