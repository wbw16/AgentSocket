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
        run_logger: object | None = None,
    ) -> None:
        resolved_memory = memory or InMemorySessionMemory()
        resolved_middleware = middleware_chain or MiddlewareChain([])
        resolved_config = runtime_config or RuntimeConfig()
        registry = ToolRegistry(tools=tools, tracer=tracer)

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
        self._engine = NativeToolEngine(**engine_kwargs)

    def run(self, user_input: str) -> AgentRunResult:
        result = self._engine.run(user_input)
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
