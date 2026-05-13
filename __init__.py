"""AgentSocket 模块化代理系统的包初始化文件。
提供了顶层导入使用的主要对象及类型声明。
"""

import sys
from pathlib import Path

# 确保 AgentSocket 目录本身在 sys.path 中，使 agent.* 的 shim 导入能正确解析
_SELF_DIR = str(Path(__file__).parent)
if _SELF_DIR not in sys.path:
    sys.path.insert(0, _SELF_DIR)

from agent.agent import Agent
from agent.clients import AnthropicClient, OpenAIClient, from_env
from agent.core.memory import InMemorySessionMemory
from agent.core.middleware import MiddlewareChain
from agent.core.run_logger import JsonlRunLogger
from agent.core.tools import ToolRegistry
from agent.core.types import (
    AgentRunResult,
    MemoryBackend,
    MemoryRecall,
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
    "JsonlRunLogger",
    "MiddlewareChain",
    "ToolRegistry",
    "AgentRunResult",
    "MemoryBackend",
    "MemoryRecall",
    "RuntimeConfig",
    "ToolCallTrace",
    "ToolCallTracer",
    "ToolSpec",
]
