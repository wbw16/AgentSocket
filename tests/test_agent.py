from unittest.mock import MagicMock

from agent.agent import Agent
from agent.core.memory import InMemorySessionMemory
from agent.core.types import NativeModelResponse, RuntimeConfig


def _make_client(final_answer="done"):
    client = MagicMock()
    client.call_with_tools.return_value = NativeModelResponse(
        final_answer=final_answer,
        tool_calls=[],
        raw_assistant_message={"role": "assistant", "content": final_answer},
    )
    return client


def test_agent_run_returns_result():
    client = _make_client("hello")
    agent = Agent(model_client=client, tools=[])
    result = agent.run("hi")
    assert result.final_answer == "hello"
    assert result.stop_reason == "finished"


def test_agent_defaults_are_applied():
    """Agent constructs defaults when optional args are None."""
    client = _make_client()
    agent = Agent(model_client=client, tools=[])
    assert agent._engine is not None


def test_agent_custom_system_prompt():
    client = _make_client()
    agent = Agent(model_client=client, tools=[], system_prompt="You are a pirate.")
    result = agent.run("hello")
    client.call_with_tools.call_args
    assert result.final_answer == "done"


def test_agent_custom_runtime_config():
    client = _make_client()
    config = RuntimeConfig(max_steps=3)
    agent = Agent(model_client=client, tools=[], runtime_config=config)
    assert agent._engine.runtime_config.max_steps == 3


def test_agent_custom_memory():
    client = _make_client()
    memory = InMemorySessionMemory()
    agent = Agent(model_client=client, tools=[], memory=memory)
    agent.run("test")
    assert len(memory.messages()) == 1
