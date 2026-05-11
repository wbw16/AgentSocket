"""Native tool-calling engine.

Uses the model API's built-in tool calling (OpenAI function calling /
Anthropic tool use) instead of the ReAct text-parsing approach.

Drop-in replacement for AgentEngine.run() — returns the same AgentRunResult.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent.core.middleware import MiddlewareChain
from agent.core.tools import ToolRegistry
from agent.core.types import AgentRunResult, MemoryBackend, ParsedStep, RuntimeConfig, ToolCallRecord, ToolResult

if TYPE_CHECKING:
    from agent.clients import AnthropicModelClient, OpenAIModelClient


def _coerce_action_input(value: Any) -> dict[str, Any]:
    """Best-effort normalize model tool arguments into an object-like dict."""
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    try:
        normalized = dict(value)
    except (TypeError, ValueError):
        return {}
    return normalized if isinstance(normalized, dict) else {}


@dataclass(slots=True)
class NativeToolEngine:
    """Runs the agent loop using the model's native tool calling API.

    The message history (`_raw_messages`) is kept in the provider's native
    format so it can be passed directly to the API each turn — no text parsing
    required.

    For Anthropic: pass `system_prompt` to set the system message (Anthropic
    keeps it separate from the messages list).
    For OpenAI-compatible: `system_prompt` is prepended as a system message.
    """

    model_client: Any  # OpenAIModelClient | AnthropicModelClient
    tool_registry: ToolRegistry
    memory: MemoryBackend
    middleware_chain: MiddlewareChain
    runtime_config: RuntimeConfig
    system_prompt: str = (
        "You are a helpful assistant. Use the provided tools to complete tasks. "
        "When you have a final answer, respond in plain text without calling any tool."
    )
    _raw_messages: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> AgentRunResult:
        self._raw_messages = self._initial_messages(user_input)
        self.memory.append_message("user", user_input)

        steps: list[ParsedStep] = []
        session_id = str(uuid.uuid4())
        started = time.time()
        tools = list(self.tool_registry.tools)

        for _ in range(self.runtime_config.max_steps):
            response = self._call_model(tools)

            if response.final_answer is not None:
                return AgentRunResult(
                    session_id=session_id,
                    final_answer=response.final_answer,
                    steps=steps,
                    action_history=self.memory.action_history(),
                    stop_reason="finished",
                    metrics={"duration_seconds": time.time() - started},
                )

            if not response.tool_calls:
                return AgentRunResult(
                    session_id=session_id,
                    final_answer="",
                    steps=steps,
                    action_history=self.memory.action_history(),
                    stop_reason="no_tool_calls",
                    metrics={"duration_seconds": time.time() - started},
                )

            # Add assistant message to history before executing tools
            self._raw_messages.append(response.raw_assistant_message)

            # Execute each tool call; collect results for multi-turn history
            pending_results: list[tuple[str, str, str]] = []  # (call_id, tool_name, result_str)

            for tc in response.tool_calls:
                try:
                    tool = self.tool_registry.get(tc.name)
                except KeyError:
                    pending_results.append((tc.call_id, tc.name, f"Error: unknown tool '{tc.name}'"))
                    continue

                action_input = _coerce_action_input(tc.arguments)

                # Middleware pre-check
                decision = self.middleware_chain.before_tool_call(
                    tool,
                    ParsedStep(thought="", action=tc.name, action_input=action_input),
                    action_input,
                )
                if decision.action in {"deny", "escalate"}:
                    return AgentRunResult(
                        session_id=session_id,
                        final_answer="",
                        steps=steps,
                        action_history=self.memory.action_history(),
                        stop_reason="denied" if decision.action == "deny" else "escalated",
                        metrics={"duration_seconds": time.time() - started, "reason": decision.reason},
                    )

                if decision.rewritten_input is not None:
                    action_input = _coerce_action_input(decision.rewritten_input)

                # Isolate handler-side mutations from recorded action_input.
                raw_result = tool.handler(dict(action_input))
                result = ToolResult(name=tool.name, output=raw_result, summary=str(raw_result))
                result = self.middleware_chain.after_tool_call(
                    tool,
                    ParsedStep(thought="", action=tc.name, action_input=action_input),
                    result,
                )

                self.memory.append_action(
                    ToolCallRecord(
                        tool=tool.name,
                        parameters=action_input,
                        result_summary=result.summary,
                    )
                )
                pending_results.append((tc.call_id, tc.name, result.summary))

            # Append tool results to history in provider-correct format
            self._append_tool_results(pending_results)

        return AgentRunResult(
            session_id=session_id,
            final_answer="",
            steps=steps,
            action_history=self.memory.action_history(),
            stop_reason="max_steps_exceeded",
            metrics={"duration_seconds": time.time() - started},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_anthropic(self) -> bool:
        return type(self.model_client).__name__ == "AnthropicModelClient"

    def _initial_messages(self, user_input: str) -> list[dict]:
        if self._is_anthropic():
            # Anthropic: system is passed separately; messages start with user
            return [{"role": "user", "content": user_input}]
        else:
            # OpenAI: system message prepended
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input},
            ]

    def _call_model(self, tools):
        if self._is_anthropic():
            return self.model_client.call_with_tools(
                self._raw_messages,
                tools,
                system=self.system_prompt,
            )
        return self.model_client.call_with_tools(self._raw_messages, tools)

    def _append_tool_results(self, results: list[tuple[str, str, str]]) -> None:
        """Add tool results to history in the provider-correct format.

        OpenAI: one {"role": "tool", ...} message per result.
        Anthropic: all results bundled into a single user message with a list of tool_result blocks.
        """
        if self._is_anthropic():
            blocks = [
                {"type": "tool_result", "tool_use_id": call_id, "content": result_str}
                for call_id, _, result_str in results
            ]
            self._raw_messages.append({"role": "user", "content": blocks})
        else:
            for call_id, tool_name, result_str in results:
                self._raw_messages.append(
                    self.model_client.tool_result_message(call_id, tool_name, result_str)
                )
