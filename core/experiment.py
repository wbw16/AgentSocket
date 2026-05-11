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
    """
    实验配置类。
    用于定义一项自动化测评或实验的各项构成参数。
    """
    name: str                                       # 实验的名称/标识
    memory_factory: Callable[[], MemoryBackend]       # 每次运行时产生全新记忆实例的工厂函数
    tools: list[ToolSpec]                           # 在此实验下代理可用的工具列表
    model_client: object                            # 语言模型客户端
    inputs: list[str]                               # 一组供代理运行的用户输入/测试问题
    runtime_config: RuntimeConfig = field(default_factory=RuntimeConfig)  # Agent执行时的参数


class ExperimentHarness:
    """
    实验基座/执行框架。
    可批量使用给定配置运行多次测试，并归总结果。
    """
    def __init__(self, run_logger: object | None = None) -> None:
        self.run_logger = run_logger   # 实验追踪器，记录所有的运行状况

    def run(self, configs: list[ExperimentConfig]) -> dict[str, list[AgentRunResult]]:
        """
        开始运行一组实验配置。
        返回结果将是所跑所有实验名称与每条用户输入跑出结果的映射字典。
        """
        results: dict[str, list[AgentRunResult]] = {}
        for config in configs:
            run_results: list[AgentRunResult] = []
            for user_input in config.inputs:
                memory = config.memory_factory()
                registry = ToolRegistry(tools=config.tools)
                # 使用该组参数初始化Agent核心执行引擎
                engine = NativeToolEngine(
                    model_client=config.model_client,
                    tool_registry=registry,
                    memory=memory,
                    middleware_chain=MiddlewareChain(middlewares=[]),
                    runtime_config=config.runtime_config,
                )
                
                # 跑对应的问题
                result = engine.run(user_input)
                
                # 若配置了logger，对此次运行情况落盘记录
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
