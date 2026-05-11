from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agent.core.types import MiddlewareDecision, ParsedStep, ToolResult, ToolSpec


class ToolMiddleware(Protocol):
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
    middlewares: list[ToolMiddleware]

    def before_tool_call(
        self,
        tool: ToolSpec,
        step: ParsedStep,
        action_input: dict[str, Any],
    ) -> MiddlewareDecision:
        current_input = dict(action_input)
        for middleware in self.middlewares:
            decision = middleware.before_tool_call(tool, step, current_input)
            if decision.action == "rewrite" and decision.rewritten_input is not None:
                current_input = dict(decision.rewritten_input)
                continue
            if decision.action != "allow":
                return decision
        return MiddlewareDecision(action="allow", rewritten_input=current_input)

    def after_tool_call(
        self,
        tool: ToolSpec,
        step: ParsedStep,
        result: ToolResult,
    ) -> ToolResult:
        current_result = result
        for middleware in self.middlewares:
            current_result = middleware.after_tool_call(tool, step, current_result)
        return current_result
