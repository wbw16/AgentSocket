"""Model client implementations for OpenAI-compatible and Anthropic APIs.

Environment variables
---------------------
Common:
  LLM_PROVIDER      "openai" | "anthropic"  (default: anthropic)
  AGENT_MODEL       model name              (default: provider-specific)

OpenAI-compatible (OpenAI, vLLM, Ollama, DeepSeek, …):
  OPENAI_API_KEY    required
  OPENAI_BASE_URL   optional (default: https://api.openai.com/v1)
  AGENT_MODEL       default: gpt-4o

Anthropic:
  ANTHROPIC_API_KEY  required
  ANTHROPIC_BASE_URL optional (default: Anthropic SDK default)
  AGENT_MODEL        default: claude-opus-4-6
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Message normalization helpers
# ---------------------------------------------------------------------------

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

    # Support inline comments in unquoted values:
    # OPENAI_BASE_URL=https://...  # comment
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


def _normalize_for_openai(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Remap 'tool' role to 'user' with a prefix; leave everything else."""
    normalized = []
    for msg in messages:
        if msg["role"] == "tool":
            normalized.append({"role": "user", "content": f"[Tool Result] {msg['content']}"})
        else:
            normalized.append(dict(msg))
    return normalized


def _normalize_for_anthropic(
    messages: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    """Split system messages out; remap/merge the rest for Anthropic's API.

    Returns (system_text, conversation_messages) where conversation_messages
    alternates user/assistant and starts with 'user'.
    """
    system_parts: list[str] = []
    convo: list[dict[str, str]] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_parts.append(content)
            continue

        # Remap 'tool' results to 'user' with a readable prefix
        effective_role = "user" if role == "tool" else role
        effective_content = f"[Tool Result] {content}" if role == "tool" else content

        # Merge consecutive messages of the same effective role
        if convo and convo[-1]["role"] == effective_role:
            convo[-1] = {
                "role": effective_role,
                "content": convo[-1]["content"] + "\n\n" + effective_content,
            }
        else:
            convo.append({"role": effective_role, "content": effective_content})

    # Anthropic requires the conversation to start with 'user'
    if convo and convo[0]["role"] != "user":
        convo.insert(0, {"role": "user", "content": "(start)"})

    # Anthropic requires the conversation to end with 'user'
    if convo and convo[-1]["role"] == "assistant":
        convo.append({"role": "user", "content": "Continue."})

    system_text = "\n\n".join(system_parts)
    return system_text, convo


def _value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------

class OpenAIModelClient:
    """Talks to any OpenAI-compatible chat API.

    Works with OpenAI, Azure OpenAI, vLLM, Ollama, DeepSeek, etc.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        capture_reasoning: bool = False,
        **extra_kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required: pip install openai"
            ) from exc

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        resolved_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        self.model = model or os.environ.get("AGENT_MODEL", "gpt-4o")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.capture_reasoning = capture_reasoning
        self.extra_kwargs = extra_kwargs
        self._client = OpenAI(api_key=resolved_key, base_url=resolved_url)

    def generate(self, messages: list[dict[str, str]]) -> str:
        if self.capture_reasoning:
            blocks = self.generate_with_blocks(messages)
            text_parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            return "\n".join(text_parts) if text_parts else ""

        normalized = _normalize_for_openai(messages)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=normalized,  # type: ignore[arg-type]
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            **self.extra_kwargs,
        )
        return response.choices[0].message.content or ""

    def generate_with_blocks(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Generate and return content blocks.

        Standard OpenAI chat responses expose only text content. Some
        OpenAI-compatible providers, notably Bailian/DashScope Qwen thinking
        models, stream private reasoning as delta.reasoning_content. When
        capture_reasoning is enabled, collect that provider extension.
        """
        if self.capture_reasoning:
            return self._generate_streaming_reasoning_blocks(messages)
        return [{"type": "text", "text": self.generate(messages)}]

    def _generate_streaming_reasoning_blocks(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized = _normalize_for_openai(messages)
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=normalized,  # type: ignore[arg-type]
            stream=True,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            **self.extra_kwargs,
        )
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for chunk in stream:
            choices = _value(chunk, "choices") or []
            if not choices:
                continue
            delta = _value(choices[0], "delta")
            if delta is None:
                continue
            reasoning = _value(delta, "reasoning_content")
            content = _value(delta, "content")
            if reasoning:
                reasoning_parts.append(str(reasoning))
            if content:
                content_parts.append(str(content))

        blocks: list[dict[str, str]] = []
        reasoning_text = "".join(reasoning_parts)
        if reasoning_text:
            blocks.append({"type": "thinking", "thinking": reasoning_text})
        blocks.append({"type": "text", "text": "".join(content_parts)})
        return blocks

    def call_with_tools(
        self,
        messages: list[dict],
        tools: list,  # list[ToolSpec]
    ) -> "NativeModelResponse":
        """Native tool calling for OpenAI-compatible APIs.

        `messages` is the raw history (OpenAI format), managed by NativeToolEngine.
        """
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
        """Format a tool result for OpenAI's multi-turn history."""
        return {"role": "tool", "tool_call_id": call_id, "content": result}


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

class AnthropicModelClient:
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
            raise ImportError(
                "anthropic package is required: pip install anthropic"
            ) from exc

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        resolved_url = base_url or os.environ.get("ANTHROPIC_BASE_URL") or None
        self.model = model or os.environ.get("AGENT_MODEL", "claude-opus-4-6")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs
        kwargs: dict[str, Any] = {"api_key": resolved_key}
        if resolved_url:
            kwargs["base_url"] = resolved_url
        self._client = _anthropic.Anthropic(**kwargs)

    def generate(self, messages: list[dict[str, str]]) -> str:
        blocks = self.generate_with_blocks(messages)
        text_parts = [b["text"] for b in blocks if b.get("type") == "text"]
        return "\n".join(text_parts) if text_parts else ""

    def generate_with_blocks(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Generate and return all content blocks including thinking."""
        system_text, convo = _normalize_for_anthropic(messages)
        if not convo:
            convo = [{"role": "user", "content": "(start)"}]

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": convo,
            **self.extra_kwargs,
        }
        if system_text:
            create_kwargs["system"] = system_text

        response = self._client.messages.create(**create_kwargs)
        blocks: list[dict[str, str]] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "thinking":
                blocks.append({"type": "thinking", "thinking": getattr(block, "thinking", "")})
            elif block_type == "text":
                blocks.append({"type": "text", "text": getattr(block, "text", "")})
        return blocks or [{"type": "text", "text": ""}]

    def call_with_tools(
        self,
        messages: list[dict],
        tools: list,  # list[ToolSpec]
        system: str = "",
    ) -> "NativeModelResponse":
        """Native tool calling for the Anthropic Messages API.

        `messages` is the raw history (Anthropic format), managed by NativeToolEngine.
        `system` is the system prompt string (separate from messages in Anthropic's API).
        """
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
            # Anthropic assistant message content is a list of blocks
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
        """Format tool results as a single user message (Anthropic's multi-turn format).

        For multiple tool calls, NativeToolEngine collects results and passes them
        together — see NativeToolEngine._append_tool_results().
        """
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": result}],
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def from_env(
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> OpenAIModelClient | AnthropicModelClient:
    """Instantiate the right client based on LLM_PROVIDER env var.

    LLM_PROVIDER=openai   → OpenAIModelClient
    LLM_PROVIDER=anthropic (default) → AnthropicModelClient
    """
    _load_dotenv_if_present()
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        return OpenAIModelClient(max_tokens=max_tokens, temperature=temperature)
    if provider == "anthropic":
        return AnthropicModelClient(max_tokens=max_tokens, temperature=temperature)
    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Expected 'openai' or 'anthropic'."
    )
