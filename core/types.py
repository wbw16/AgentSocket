from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol


JSONDict = dict[str, Any]


class ModelClient(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(slots=True)
class IntentPayload:
    action_type: str
    target_resource: str
    destination: str
    intent_basis: str
    side_effect: str


@dataclass(slots=True)
class ParsedStep:
    thought: str
    action: str | None = None
    action_input: JSONDict = field(default_factory=dict)
    intent: IntentPayload | None = None
    final_answer: str | None = None
    raw_output: str = ""


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[[JSONDict], Any]
    requires_intent: bool = False
    middlewares: tuple[str, ...] = ()
    risk_level: str = "normal"
    # JSON Schema describing the tool's parameters.
    # When set, the engine can use the API's native tool calling instead of ReAct parsing.
    # Example: {"type": "object", "properties": {"to": {"type": "string"}}, "required": ["to"]}
    parameters_schema: JSONDict = field(default_factory=dict)


@dataclass(slots=True)
class NativeToolCall:
    """A single tool call returned by the model via native tool calling."""
    call_id: str
    name: str
    arguments: JSONDict


@dataclass(slots=True)
class NativeModelResponse:
    """Structured response from a native tool-calling API turn."""
    final_answer: str | None          # set when model is done (no tool calls)
    tool_calls: list[NativeToolCall]  # set when model wants to call tools
    raw_assistant_message: dict       # the exact assistant message to add to history


@dataclass(slots=True)
class ToolCallRecord:
    tool: str
    parameters: JSONDict
    result_summary: str
    intent: IntentPayload | None = None


@dataclass(slots=True)
class ToolResult:
    name: str
    output: Any
    summary: str


@dataclass(slots=True)
class MiddlewareDecision:
    action: str
    reason: str = ""
    rewritten_input: JSONDict | None = None
    observation_override: Any | None = None


@dataclass(slots=True)
class AgentState:
    session_id: str
    user_input: str
    step_index: int = 0
    finished: bool = False


@dataclass(slots=True)
class AgentRunResult:
    session_id: str
    final_answer: str
    steps: list[ParsedStep]
    action_history: list[ToolCallRecord]
    stop_reason: str
    metrics: Mapping[str, Any]


@dataclass(slots=True)
class ToolCallTrace:
    session_id: str
    step_index: int
    tool_name: str
    arguments: dict
    result_summary: str
    duration_ms: float
    middleware_decision: str  # "allow" | "deny" | "rewrite" | "escalate"
    timestamp: float


class ToolCallTracer(Protocol):
    def record(self, trace: ToolCallTrace) -> None: ...


class MemoryBackend(Protocol):
    def append_message(self, role: str, content: str) -> None: ...
    def append_action(self, record: ToolCallRecord) -> None: ...
    def messages(self) -> list[dict[str, str]]: ...
    def action_history(self) -> list[ToolCallRecord]: ...
    def retrieve(self, query: str, k: int = 5) -> list[dict]: ...
    def summarize(self, max_tokens: int) -> str: ...


@dataclass(slots=True)
class RuntimeConfig:
    max_steps: int = 8
    max_correction_retries: int = 2
    stop_on_tool_error: bool = True
