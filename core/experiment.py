from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable

from agent.core.engine_native import NativeToolEngine
from agent.core.middleware import MiddlewareChain
from agent.core.tools import ToolRegistry
from agent.core.types import AgentRunResult, MemoryBackend, RuntimeConfig, ToolSpec


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    memory_factory: Callable[[], MemoryBackend]
    tools: list[ToolSpec]
    model_client: object
    inputs: list[str]
    runtime_config: RuntimeConfig = field(default_factory=RuntimeConfig)


class ExperimentHarness:
    def run(self, configs: list[ExperimentConfig]) -> dict[str, list[AgentRunResult]]:
        results: dict[str, list[AgentRunResult]] = {}
        for config in configs:
            run_results: list[AgentRunResult] = []
            for user_input in config.inputs:
                memory = config.memory_factory()
                registry = ToolRegistry(tools=config.tools)
                engine = NativeToolEngine(
                    model_client=config.model_client,
                    tool_registry=registry,
                    memory=memory,
                    middleware_chain=MiddlewareChain(middlewares=[]),
                    runtime_config=config.runtime_config,
                )
                run_results.append(engine.run(user_input))
            results[config.name] = run_results
        return results
