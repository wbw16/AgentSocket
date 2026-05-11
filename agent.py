from __future__ import annotations

from pathlib import Path

if __name__ == "agent":
    __path__ = [str(Path(__file__).parent)]  # type: ignore[name-defined]

from agent.core.engine_native import NativeToolEngine
from agent.core.memory import InMemorySessionMemory
from agent.core.middleware import MiddlewareChain
from agent.core.tools import ToolRegistry
from agent.core.types import (
    AgentRunResult,
    MemoryBackend,
    RuntimeConfig,
    ToolCallTracer,
    ToolSpec,
)


class Agent:
    """Agent框架的主要测试/运行入口。

    大型语言模型（LLM）的具体配置交由 model_client（例如 AnthropicClient 或 OpenAIClient）来管理。
    所有其他模块和组件都是可配置且非必须的；如果未提供，系统会采用默认合理的无操作（no-op）实现或基础实现。
    """

    def __init__(
        self,
        model_client: object,                           # 语言模型客户端对象
        tools: list[ToolSpec],                          # 该 Agent 可用的工具清单
        memory: MemoryBackend | None = None,            # 记忆后端存储（默认使用基于内存的 InMemorySessionMemory）
        middleware_chain: MiddlewareChain | None = None,# 中间件拦截器链（默认空）
        runtime_config: RuntimeConfig | None = None,    # 运行时配置（默认基础配置）
        system_prompt: str | None = None,               # 系统提示词（可选）
        tracer: ToolCallTracer | None = None,           # 追踪记录器，记录工具的调用过程
        run_logger: object | None = None,               # 执行过程记录器
    ) -> None:
        # 若未提供参数则应用默认基础实现
        resolved_memory = memory or InMemorySessionMemory()
        resolved_middleware = middleware_chain or MiddlewareChain([])
        resolved_config = runtime_config or RuntimeConfig()
        registry = ToolRegistry(tools=tools, tracer=tracer)

        # 组装传给原生引擎的参数列表
        engine_kwargs: dict = {
            "model_client": model_client,
            "tool_registry": registry,
            "memory": resolved_memory,
            "middleware_chain": resolved_middleware,
            "runtime_config": resolved_config,
        }
        if system_prompt is not None:
            engine_kwargs["system_prompt"] = system_prompt

        self._memory = resolved_memory
        self._run_logger = run_logger
        # 实例化负责真正推演的原生工具执行引擎
        self._engine = NativeToolEngine(**engine_kwargs)

    def run(self, user_input: str) -> AgentRunResult:
        """接收用户的文本输入，触发一轮推演并返回执行结果。"""
        result = self._engine.run(user_input)
        # 如果存在执行日志器，记录本次代理运行信息及结果
        if self._run_logger is not None:
            self._run_logger.record_run(
                user_input=user_input,
                result=result,
                memory=self._memory,
            )
        return result


if __name__ == "agent":
    from agent.clients import AnthropicClient, OpenAIClient, from_env
    from agent.core.memory import InMemorySessionMemory
    from agent.core.run_logger import JsonlRunLogger
    from agent.core.types import ToolCallTrace

    # 定义模块公开导出的相关类和方法
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
        "RuntimeConfig",
        "ToolCallTrace",
        "ToolCallTracer",
        "ToolSpec",
    ]
