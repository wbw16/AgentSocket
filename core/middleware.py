from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agent.core.types import MiddlewareDecision, ParsedStep, ToolResult, ToolSpec


class ToolMiddleware(Protocol):
    """
    代理工具拦截/处理中间件协议。
    中间件支持在工具执行前与执行后对入参、决定及结果进行风控与拦截修改操作。
    """
    def before_tool_call(
        self,
        tool: ToolSpec,
        step: ParsedStep,
        action_input: dict[str, Any],
    ) -> MiddlewareDecision:
        ...

    def after_tool_call(
        self,
        tool: ToolSpec,
        step: ParsedStep,
        result: ToolResult,
    ) -> ToolResult:
        ...


@dataclass(slots=True)
class MiddlewareChain:
    """
    中间件责任执行链。
    按顺序组合并触发多个 Middleware 来共同控制工具的调用安全及副作用。
    """
    middlewares: list[ToolMiddleware]

    def before_tool_call(
        self,
        tool: ToolSpec,
        step: ParsedStep,
        action_input: dict[str, Any],
    ) -> MiddlewareDecision:
        """
        在调用真正的工具逻辑之前验证权限以及是否允许调用。
        如果有任何一个中间件返回非 'allow' 或 'rewrite' 外的拦截指令，校验马上停止，且抛出决策结果。
        """
        current_input = dict(action_input)
        for middleware in self.middlewares:
            decision = middleware.before_tool_call(tool, step, current_input)
            # 处理希望修改重写模型传入参数的情况
            if decision.action == "rewrite" and decision.rewritten_input is not None:
                current_input = dict(decision.rewritten_input)
                continue
            # 处理要求阻断或需要人为干预的情况
            if decision.action != "allow":
                return decision
        return MiddlewareDecision(action="allow", rewritten_input=current_input)

    def after_tool_call(
        self,
        tool: ToolSpec,
        step: ParsedStep,
        result: ToolResult,
    ) -> ToolResult:
        """
        在工具执行完毕拿到了原始数据后对其进行脱敏、检查或是二次包装加工处理。
        """
        current_result = result
        for middleware in self.middlewares:
            current_result = middleware.after_tool_call(tool, step, current_result)
        return current_result
