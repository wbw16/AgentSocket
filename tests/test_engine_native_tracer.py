from dataclasses import dataclass, field

from agent.core.engine_native import NativeToolEngine
from agent.core.memory import InMemorySessionMemory
from agent.core.middleware import MiddlewareChain
from agent.core.tools import ToolRegistry
from agent.core.types import (
    NativeModelResponse,
    NativeToolCall,
    RuntimeConfig,
    ToolCallTrace,
    ToolSpec,
)


@dataclass
class CapturingTracer:
    records: list[ToolCallTrace] = field(default_factory=list)

    def record(self, trace: ToolCallTrace) -> None:
        self.records.append(trace)


class FakeClient:
    """Returns one tool call then a final answer."""

    def __init__(self):
        self._call_count = 0

    def call_with_tools(self, messages, tools, system=""):
        self._call_count += 1
        if self._call_count == 1:
            return NativeModelResponse(
                final_answer=None,
                tool_calls=[NativeToolCall(call_id="c1", name="echo", arguments={"msg": "hi"})],
                raw_assistant_message={"role": "assistant", "content": []},
            )
        return NativeModelResponse(
            final_answer="done",
            tool_calls=[],
            raw_assistant_message={"role": "assistant", "content": "done"},
        )

    def tool_result_message(self, call_id, tool_name, result):
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": result}],
        }


def test_tracer_records_tool_call():
    tracer = CapturingTracer()
    tool = ToolSpec(
        name="echo",
        description="echo",
        handler=lambda args: args["msg"],
        parameters_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
    )
    registry = ToolRegistry(tools=[tool], tracer=tracer)
    engine = NativeToolEngine(
        model_client=FakeClient(),
        tool_registry=registry,
        memory=InMemorySessionMemory(),
        middleware_chain=MiddlewareChain([]),
        runtime_config=RuntimeConfig(),
    )
    result = engine.run("say hi")
    assert result.final_answer == "done"
    assert len(tracer.records) == 1
    trace = tracer.records[0]
    assert trace.tool_name == "echo"
    assert trace.arguments == {"msg": "hi"}
    assert trace.result_summary == "hi"
    assert trace.middleware_decision == "allow"
    assert trace.duration_ms >= 0
    assert trace.timestamp > 0


def test_tracer_none_no_error():
    """No tracer set: engine runs without error."""
    tool = ToolSpec(
        name="echo",
        description="echo",
        handler=lambda args: args["msg"],
        parameters_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
    )
    registry = ToolRegistry(tools=[tool])
    engine = NativeToolEngine(
        model_client=FakeClient(),
        tool_registry=registry,
        memory=InMemorySessionMemory(),
        middleware_chain=MiddlewareChain([]),
        runtime_config=RuntimeConfig(),
    )
    result = engine.run("say hi")
    assert result.final_answer == "done"
