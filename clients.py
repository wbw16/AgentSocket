"""Model client implementations for OpenAI-compatible and Anthropic APIs.
(用于兼容 OpenAI 和 Anthropic API 的各大语言模型客户端实现代码)

环境变量 (Environment variables)
---------------------
公共变量 (Common):
  LLM_PROVIDER      "openai" | "anthropic"  (默认：anthropic)
  AGENT_MODEL       模型名称              (默认：按提供商标定)

OpenAI 兼容平台 (如 OpenAI, vLLM, Ollama, DeepSeek 等):
  OPENAI_API_KEY    必须提供
  OPENAI_BASE_URL   可选 (默认值: https://api.openai.com/v1)
  AGENT_MODEL       默认值: gpt-4o

Anthropic 平台:
  ANTHROPIC_API_KEY 必须提供
  ANTHROPIC_BASE_URL可选 (默认值为 Anthropic SDK 默认地址)
  AGENT_MODEL       默认值: claude-opus-4-7
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    """逐行解析 .env 文件内容，提取出环境变量的键值对。"""
    line = line.strip()
    # 忽略空行或注释行
    if not line or line.startswith("#"):
        return None
    # 移除 'export ' 前缀（如果有）
    if line.startswith("export "):
        line = line[len("export "):].lstrip()
    if "=" not in line:
        return None
    key, raw_value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = raw_value.strip()
    # 移除被包含的引号
    if (
        len(value) >= 2
        and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"))
    ):
        return key, value[1:-1]
    # 清理内联注释
    value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
    return key, value


def _load_dotenv_if_present(path: Path | None = None) -> None:
    """如果存在 .env 文件，则加载并将里面的变量赋值到当前 os.environ 中。"""
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
    """安全地从字典获取键值，或从对象获取属性的方法。"""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


class OpenAIClient:
    """与任何兼容 OpenAI 的聊天 API (比如 OpenAI 本身, vLLM, Ollama, DeepSeek 等) 进行通讯的客户端类。"""

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
        # 尝试引入 openai 包，未安装时给出提示
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is required: pip install openai") from exc

        # 优先使用显式传入的参数，否则尝试读取环境变量
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        resolved_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        self.model = model or os.environ.get("AGENT_MODEL", "gpt-4o")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs
        # 通过初始化的 API 配置参数创建 OpenAI 客户端
        self._client = OpenAI(api_key=resolved_key, base_url=resolved_url)

    def call_with_tools(
        self,
        messages: list[dict],
        tools: list,
    ) -> "NativeModelResponse":
        """携带工具信息并按 OpenAI 的请求格式发起模型请求计算"""
        from agent.core.types import NativeModelResponse, NativeToolCall
        import json

        # 将业务侧定义的 tools 转换成 OpenAI function calling 需要的格式
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

        # 调用接口进行推理
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            tools=openai_tools,  # type: ignore[arg-type]
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            **self.extra_kwargs,
        )

        msg = response.choices[0].message

        # 如果返回体内带有工具调用的命令 (tool_calls)
        if msg.tool_calls:
            # 整理为通用的 NativeToolCall 结构
            native_calls = [
                NativeToolCall(
                    call_id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in msg.tool_calls
            ]
            # 保留 OpenAI 在下次通讯时要求的响应格式
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
            # 处理具备“推理内容(reasoning_content)”返回的模型情况
            reasoning_content = _value(msg, "reasoning_content")
            if reasoning_content is not None:
                raw_assistant["reasoning_content"] = reasoning_content
                
            return NativeModelResponse(
                final_answer=None,
                tool_calls=native_calls,
                raw_assistant_message=raw_assistant,
            )

        # 走到这里意味着模型完成了推理并给出了最终文本回答
        return NativeModelResponse(
            final_answer=msg.content or "",
            tool_calls=[],
            raw_assistant_message={"role": "assistant", "content": msg.content or ""},
        )

    def tool_result_message(self, call_id: str, _tool_name: str, result: str) -> dict:
        """用于组织格式化的包含了工具执行结果的历史消息，回传给 OpenAI 兼容 API。"""
        return {"role": "tool", "tool_call_id": call_id, "content": result}


class AnthropicClient:
    """与 Anthropic Messages API 进行通讯的客户端类。"""

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
        """携带工具列表向 Anthropic 请求文本补全"""
        from agent.core.types import NativeModelResponse, NativeToolCall

        # 将工具统一转换为 Anthropic 的输入格式
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
        # 如果存在系统消息，加入到专门的参数系统级别指令中
        if system:
            create_kwargs["system"] = system

        response = self._client.messages.create(**create_kwargs)

        # 筛选返回区块里的不同种类内容
        tool_use_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        text_blocks = [b for b in response.content if getattr(b, "type", None) == "text"]

        # 发现工具指派的命令，提取执行参数
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
            # 混入如果有的话一起带上的辅助纯文本推理块
            if text_blocks:
                raw_content = [{"type": "text", "text": text_blocks[0].text}] + raw_content
                
            return NativeModelResponse(
                final_answer=None,
                tool_calls=native_calls,
                raw_assistant_message={"role": "assistant", "content": raw_content},
            )

        # 全为纯文本块意味着推理已经完成并输出目标信息结果
        text = text_blocks[0].text if text_blocks else ""
        return NativeModelResponse(
            final_answer=text,
            tool_calls=[],
            raw_assistant_message={"role": "assistant", "content": text},
        )

    def tool_result_message(self, call_id: str, _tool_name: str, result: str) -> dict:
        """用于组装符合 Anthropic API 要求包含工具结果的对象，放于下一轮对话 User 身份角色中。"""
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": result}],
        }


def from_env(
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> OpenAIClient | AnthropicClient:
    """根据运行时环境标志 `LLM_PROVIDER` 解析并实例化适配的对应大语言模型客户端。"""
    _load_dotenv_if_present()
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        return OpenAIClient(max_tokens=max_tokens, temperature=temperature)
    if provider == "anthropic":
        return AnthropicClient(max_tokens=max_tokens, temperature=temperature)
    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Expected 'openai' or 'anthropic'."
    )
