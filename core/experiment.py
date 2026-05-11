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
    def __init__(self, run_logger: object | None = None) -> None:
        self.run_logger = run_logger

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
                result = engine.run(user_input)
                if self.run_logger is not None:
                    self.run_logger.record_run(
                        user_input=user_input,
                        result=result,
                        memory=memory,
                        experiment_name=config.name,
                    )
                run_results.append(result)
            results[config.name] = run_results
        return results
