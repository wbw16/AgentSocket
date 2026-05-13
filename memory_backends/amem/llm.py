from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


class LLMClient(Protocol):
    """A-MEM 内部使用的 LLM 协议，与 AgentSocket 的 ModelClient 解耦。"""

    def complete(self, messages: list[dict], **kwargs) -> str:
        """发送消息列表，返回纯文本回复。"""
        ...


class AnthropicAdapter:
    """把 AgentSocket AnthropicClient 适配成 LLMClient 协议。

    A-MEM 本身不 import AgentSocket clients 模块，
    由调用方在构造 AMemBackend 时传入已实例化的 client。
    """

    def __init__(self, client: object, model: str | None = None) -> None:
        self._client = client
        # 允许覆盖 model，否则沿用 client 自身的 model 属性
        self._model = model

    def complete(self, messages: list[dict], **kwargs) -> str:
        import anthropic as _anthropic

        # 分离 system 消息（Anthropic API 要求单独传）
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        user_messages = [m for m in messages if m.get("role") != "system"]
        system_str = "\n\n".join(system_parts)

        model = self._model or getattr(self._client, "model", "claude-haiku-4-5")
        max_tokens = kwargs.get("max_tokens", 1024)
        temperature = kwargs.get("temperature", 0.3)

        raw_client = getattr(self._client, "_client", None)
        if raw_client is None:
            raise RuntimeError("AnthropicAdapter: client._client not found")

        create_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages,
        }
        if system_str:
            create_kwargs["system"] = system_str

        response = raw_client.messages.create(**create_kwargs)
        text_blocks = [b for b in response.content if getattr(b, "type", None) == "text"]
        return text_blocks[0].text if text_blocks else ""


class OpenAIAdapter:
    """把 AgentSocket OpenAIClient 适配成 LLMClient 协议。"""

    def __init__(self, client: object, model: str | None = None) -> None:
        self._client = client
        self._model = model

    def complete(self, messages: list[dict], **kwargs) -> str:
        model = self._model or getattr(self._client, "model", "gpt-4o-mini")
        max_tokens = kwargs.get("max_tokens", 1024)
        temperature = kwargs.get("temperature", 0.3)

        raw_client = getattr(self._client, "_client", None)
        if raw_client is None:
            raise RuntimeError("OpenAIAdapter: client._client not found")

        response = raw_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
