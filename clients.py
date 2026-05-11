"""Model client implementations for OpenAI-compatible and Anthropic APIs.

Environment variables
---------------------
Common:
  LLM_PROVIDER      "openai" | "anthropic"  (default: anthropic)
  AGENT_MODEL       model name              (default: provider-specific)

OpenAI-compatible (OpenAI, vLLM, Ollama, DeepSeek, ...):
  OPENAI_API_KEY    required
  OPENAI_BASE_URL   optional (default: https://api.openai.com/v1)
  AGENT_MODEL       default: gpt-4o

Anthropic:
  ANTHROPIC_API_KEY  required
  ANTHROPIC_BASE_URL optional (default: Anthropic SDK default)
  AGENT_MODEL        default: claude-opus-4-7
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export "):].lstrip()
    if "=" not in line:
        return None
    key, raw_value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = raw_value.strip()
    if (
        len(value) >= 2
        and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"))
    ):
        return key, value[1:-1]
    value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
    return key, value


def _load_dotenv_if_present(path: Path | None = None) -> None:
    dotenv_path = path or (Path.cwd() / ".env")
    if not dotenv_path.exists():
        return
    try:
        content = dotenv_path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in content.splitlines():
        parsed = _parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def _value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


class OpenAIClient:
    """Talks to any OpenAI-compatible chat API (OpenAI, vLLM, Ollama, DeepSeek, etc.)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **extra_kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is required: pip install openai") from exc

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        resolved_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        self.model = model or os.environ.get("AGENT_MODEL", "gpt-4o")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs
        self._client = OpenAI(api_key=resolved_key, base_url=resolved_url)

    def call_with_tools(
        self,
        messages: list[dict],
        tools: list,
    ) -> "NativeModelResponse":
        from agent.core.types import NativeModelResponse, NativeToolCall
        import json

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            tools=openai_tools,  # type: ignore[arg-type]
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            **self.extra_kwargs,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            native_calls = [
                NativeToolCall(
                    call_id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in msg.tool_calls
            ]
            raw_assistant = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
            return NativeModelResponse(
                final_answer=None,
                tool_calls=native_calls,
                raw_assistant_message=raw_assistant,
            )

        return NativeModelResponse(
            final_answer=msg.content or "",
            tool_calls=[],
            raw_assistant_message={"role": "assistant", "content": msg.content or ""},
        )

    def tool_result_message(self, call_id: str, _tool_name: str, result: str) -> dict:
        return {"role": "tool", "tool_call_id": call_id, "content": result}


class AnthropicClient:
    """Talks to the Anthropic Messages API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **extra_kwargs: Any,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError("anthropic package is required: pip install anthropic") from exc

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        resolved_url = base_url or os.environ.get("ANTHROPIC_BASE_URL") or None
        self.model = model or os.environ.get("AGENT_MODEL", "claude-opus-4-7")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs
        kwargs: dict[str, Any] = {"api_key": resolved_key}
        if resolved_url:
            kwargs["base_url"] = resolved_url
        self._client = _anthropic.Anthropic(**kwargs)

    def call_with_tools(
        self,
        messages: list[dict],
        tools: list,
        system: str = "",
    ) -> "NativeModelResponse":
        from agent.core.types import NativeModelResponse, NativeToolCall

        anthropic_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters_schema or {"type": "object", "properties": {}},
            }
            for t in tools
        ]

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
            "tools": anthropic_tools,
            **self.extra_kwargs,
        }
        if system:
            create_kwargs["system"] = system

        response = self._client.messages.create(**create_kwargs)

        tool_use_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        text_blocks = [b for b in response.content if getattr(b, "type", None) == "text"]

        if tool_use_blocks:
            native_calls = [
                NativeToolCall(
                    call_id=b.id,
                    name=b.name,
                    arguments=dict(b.input),
                )
                for b in tool_use_blocks
            ]
            raw_content = [
                {"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)}
                for b in tool_use_blocks
            ]
            if text_blocks:
                raw_content = [{"type": "text", "text": text_blocks[0].text}] + raw_content
            return NativeModelResponse(
                final_answer=None,
                tool_calls=native_calls,
                raw_assistant_message={"role": "assistant", "content": raw_content},
            )

        text = text_blocks[0].text if text_blocks else ""
        return NativeModelResponse(
            final_answer=text,
            tool_calls=[],
            raw_assistant_message={"role": "assistant", "content": text},
        )

    def tool_result_message(self, call_id: str, _tool_name: str, result: str) -> dict:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": result}],
        }


def from_env(
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> OpenAIClient | AnthropicClient:
    """Instantiate the right client based on LLM_PROVIDER env var."""
    _load_dotenv_if_present()
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        return OpenAIClient(max_tokens=max_tokens, temperature=temperature)
    if provider == "anthropic":
        return AnthropicClient(max_tokens=max_tokens, temperature=temperature)
    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Expected 'openai' or 'anthropic'."
    )
