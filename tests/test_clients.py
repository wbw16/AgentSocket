import pytest
from types import SimpleNamespace

from agent.core.types import ToolSpec


def test_anthropic_client_importable():
    from agent.clients import AnthropicClient

    assert AnthropicClient is not None


def test_openai_client_importable():
    from agent.clients import OpenAIClient

    assert OpenAIClient is not None


def test_old_names_gone():
    import agent.clients as m

    assert not hasattr(m, "OpenAIModelClient")
    assert not hasattr(m, "AnthropicModelClient")


def test_generate_method_gone():
    from agent.clients import AnthropicClient

    assert not hasattr(AnthropicClient, "generate")
    assert not hasattr(AnthropicClient, "generate_with_blocks")


def test_call_with_tools_present():
    from agent.clients import AnthropicClient, OpenAIClient

    assert hasattr(AnthropicClient, "call_with_tools")
    assert hasattr(OpenAIClient, "call_with_tools")


def test_openai_client_preserves_reasoning_content_in_tool_message():
    from agent.clients import OpenAIClient

    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="echo", arguments='{"text": "hi"}'),
    )
    message = SimpleNamespace(
        content=None,
        reasoning_content="thinking trace",
        tool_calls=[tool_call],
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )
    client = object.__new__(OpenAIClient)
    client.model = "test-model"
    client.max_tokens = 128
    client.temperature = 0.0
    client.extra_kwargs = {}
    client._client = fake_client
    tool = ToolSpec(
        name="echo",
        description="echo",
        handler=lambda args: args["text"],
        parameters_schema={"type": "object", "properties": {"text": {"type": "string"}}},
    )

    result = client.call_with_tools(messages=[], tools=[tool])

    assert result.raw_assistant_message["reasoning_content"] == "thinking trace"
