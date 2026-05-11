import pytest


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
