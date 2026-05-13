"""原生工具调用引擎。

此引擎使用了各大模型API内置的工具调用（例如 OpenAI 的 function calling 或 Anthropic 的 tool use）
以取代传统的 ReAct 文本生成及解析模式。

作为 AgentEngine.run() 的直接替代模块，它会返回相同的 AgentRunResult 结构对象。
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent.core.middleware import MiddlewareChain
from agent.core.tools import ToolRegistry
from agent.core.types import (
    AgentRunResult,
    MemoryBackend,
    MemoryRecall,
    ParsedStep,
    RuntimeConfig,
    ToolCallRecord,
    ToolCallTrace,
    ToolResult,
)

if TYPE_CHECKING:
    from agent.clients import AnthropicClient, OpenAIClient


def _coerce_action_input(value: Any) -> dict[str, Any]:
    """尽可能地将模型输出的工具参数规范化为字典（dict）对象。"""
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
    """使用模型的原生工具调用（Native Tool Calling）API来运行代理（Agent）的执行循环。

    消息历史记录(`_raw_messages`)保留了对应供应商指定的原有格式，以便每次轮流迭代直接发送结 API，
    完全省下了解析生成文本的繁琐过程。

    对于 Anthropic: 
        通过传入 `system_prompt` 参数来设定系统提示信息 (Anthropic 的 API 中，系统消息是与消息列表分离的)，
    对于兼容 OpenAI 的 API: 
        将 `system_prompt` 自动置于首条作为单独的一条系统信息 (System Message)。
    """

    model_client: Any  # OpenAIClient | AnthropicClient 的实例
    tool_registry: ToolRegistry  # 工具注册表
    memory: MemoryBackend        # 用于记录 Agent 会话记忆的后端
    middleware_chain: MiddlewareChain  # 用于请求检查拦截的中间件责任链
    runtime_config: RuntimeConfig      # Agent 执行期的配置项
    # 系统级别的默认提示词
    system_prompt: str = (
        "You are a helpful assistant. Use the provided tools to complete tasks. "
        "When you have a final answer, respond in plain text without calling any tool."
    )
    # 保存原始的对话上下文数组格式，用于跟大模型交互
    _raw_messages: list[dict] = field(default_factory=list)
    # 当前 run 的召回内容（Anthropic 分支在 _call_model 中拼进 system_prompt 使用）
    _pending_recall_block: str = ""

    # ------------------------------------------------------------------
    # 公开接口 (Public interface)
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> AgentRunResult:
        """接收用户的文本输入，运行推理及多轮工具调用的全流程，并返回最终结果。"""
        # 写入信号：用户输入先进记忆，backend 自己决定是否落库
        self.memory.append_message("user", user_input)

        # 召回：统一在 run 开头做一次，backend 决定返回什么 / 返回多少
        recalled: list[MemoryRecall] = list(self.memory.retrieve(user_input) or [])

        # 缓存供 Anthropic 分支在 _call_model 中拼进 system_prompt
        self._pending_recall_block = self._format_recall(recalled)

        # 初始化消息记录；如有召回内容，会拼到 system 段或 user 段之前
        self._raw_messages = self._initial_messages(user_input, recalled)

        steps: list[ParsedStep] = []
        session_id = str(uuid.uuid4())
        started = time.time()
        tools = list(self.tool_registry.tools)

        # 进入主循环，上限受 runtime_config.max_steps 控制
        for _ in range(self.runtime_config.max_steps):
            response = self._call_model(tools)

            # 当返回包含 final_answer 时，说明任务结束
            if response.final_answer is not None:
                # 写入信号：最终回答入记忆，闭合 user/assistant 语义对
                self.memory.append_message("assistant", response.final_answer)
                return AgentRunResult(
                    session_id=session_id,
                    final_answer=response.final_answer,
                    steps=steps,
                    action_history=self.memory.action_history(),
                    stop_reason="finished",
                    metrics={"duration_seconds": time.time() - started},
                    memory_recall=recalled,
                )

            # 如果模型没有给出有效工具调用，也会提前结束
            if not response.tool_calls:
                return AgentRunResult(
                    session_id=session_id,
                    final_answer="",
                    steps=steps,
                    action_history=self.memory.action_history(),
                    stop_reason="no_tool_calls",
                    metrics={"duration_seconds": time.time() - started},
                    memory_recall=recalled,
                )

            # 在真正触发工具前，首先把助手的原格式消息加进对话历史
            self._raw_messages.append(response.raw_assistant_message)

            # 执行需要调用的每个工具; 把拿到的结果临时存在这里，用于最后追加至整体历史中
            pending_results: list[tuple[str, str, str]] = []  # 分别为 (call_id, 工具名称, 返回文本)

            for tc in response.tool_calls:
                try:
                    tool = self.tool_registry.get(tc.name)
                except KeyError:
                    # 碰到未注册的工具时记录错误
                    pending_results.append((tc.call_id, tc.name, f"Error: unknown tool '{tc.name}'"))
                    continue

                action_input = _coerce_action_input(tc.arguments)

                # 工具被调用前的中间件拦截检查（例如做风控校验）
                decision = self.middleware_chain.before_tool_call(
                    tool,
                    ParsedStep(thought="", action=tc.name, action_input=action_input),
                    action_input,
                )
                
                # 若被拦截或者升级决策，则直接结束本轮
                if decision.action in {"deny", "escalate"}:
                    return AgentRunResult(
                        session_id=session_id,
                        final_answer="",
                        steps=steps,
                        action_history=self.memory.action_history(),
                        stop_reason="denied" if decision.action == "deny" else "escalated",
                        metrics={"duration_seconds": time.time() - started, "reason": decision.reason},
                        memory_recall=recalled,
                    )

                # 中间件有权篡改或重写入参
                if decision.rewritten_input is not None:
                    action_input = _coerce_action_input(decision.rewritten_input)

                # 隔绝运行器中可能产生的不安全变量变动，并调用函数
                tool_start = time.time()
                raw_result = tool.handler(dict(action_input))
                result = ToolResult(name=tool.name, output=raw_result, summary=str(raw_result))
                
                # 工具调用后的再次经过中间件把关或结果重写
                result = self.middleware_chain.after_tool_call(
                    tool,
                    ParsedStep(thought="", action=tc.name, action_input=action_input),
                    result,
                )
                tool_end = time.time()

                # 如果存在 tracer，则提交一份记录
                if self.tool_registry.tracer is not None:
                    self.tool_registry.tracer.record(
                        ToolCallTrace(
                            session_id=session_id,
                            step_index=len(steps),
                            tool_name=tc.name,
                            arguments=action_input,
                            result_summary=result.summary,
                            duration_ms=(tool_end - tool_start) * 1000,
                            middleware_decision=decision.action,
                            timestamp=tool_start,
                        )
                    )

                # 留下对本次工具调用的历史状态
                self.memory.append_action(
                    ToolCallRecord(
                        tool=tool.name,
                        parameters=action_input,
                        result_summary=result.summary,
                    )
                )
                pending_results.append((tc.call_id, tc.name, result.summary))

            # 追加工具返回的结果（将统一化为对应模型的 API 的格式）
            self._append_tool_results(pending_results)

        # 超出规定的最大步数限制后结束运行
        return AgentRunResult(
            session_id=session_id,
            final_answer="",
            steps=steps,
            action_history=self.memory.action_history(),
            stop_reason="max_steps_exceeded",
            metrics={"duration_seconds": time.time() - started},
            memory_recall=recalled,
        )

    # ------------------------------------------------------------------
    # 内部助手函数 (Internal helpers)
    # ------------------------------------------------------------------

    def _is_anthropic(self) -> bool:
        """检查当前所载入的模型 API 是不是 Anthropic 种类。"""
        return type(self.model_client).__name__ == "AnthropicClient"

    @staticmethod
    def _format_recall(recalled: list[MemoryRecall]) -> str:
        """把召回内容格式化成注入 prompt 用的 <memory> 段，空召回返回空串。"""
        if not recalled:
            return ""
        lines = [f"- {item.text}" for item in recalled if item.text]
        if not lines:
            return ""
        return "<memory>\n" + "\n".join(lines) + "\n</memory>"

    def _initial_messages(self, user_input: str, recalled: list[MemoryRecall]) -> list[dict]:
        """依据提供商构建会话开局阶段的历史消息集合。

        召回注入策略固定为：
        - OpenAI 兼容：在原 system 消息之后追加一条新的 system 消息承载 <memory>
        - Anthropic：由 _call_model 把 <memory> 拼到 system_prompt 尾部
        空召回则完全等价于注入前行为。
        """
        recall_block = self._format_recall(recalled)
        if self._is_anthropic():
            # Anthropic 的 system 以参数形式下发，召回注入在 _call_model 完成
            return [{"role": "user", "content": user_input}]
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        if recall_block:
            messages.append({"role": "system", "content": recall_block})
        messages.append({"role": "user", "content": user_input})
        return messages

    def _call_model(self, tools):
        """通用模型请求入口。把工具结构一起传给相应的 LLM 客户端 API。"""
        if self._is_anthropic():
            # Anthropic 的召回块拼到 system 尾部，与 OpenAI 分支（额外 system 消息）行为对齐
            system = self.system_prompt
            if self._pending_recall_block:
                system = f"{system}\n\n{self._pending_recall_block}"
            return self.model_client.call_with_tools(
                self._raw_messages,
                tools,
                system=system,
            )
        return self.model_client.call_with_tools(self._raw_messages, tools)

    def _append_tool_results(self, results: list[tuple[str, str, str]]) -> None:
        """向当前的聊天记录中正确插入带有工具调用返回结果的上下文消息。

        OpenAI: 为每一个调用的结果加上单独的一条 {"role": "tool", ...} 格式的日志。
        Anthropic: 它们需要统一被组合进同一条属于 "user" 的消息结构里，通过 type 等于 tool_result 标出。
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
