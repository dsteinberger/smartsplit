"""OpenAI ↔ Anthropic Messages API protocol adapter.

Bidirectional translation between OpenAI chat completions format and
Anthropic Messages API format, covering messages, system prompts, tools,
tool_calls, tool_results, and usage mapping.

Used both for direct ``AnthropicProvider.complete`` calls and for the
``ProxyRegistry.proxy_to_brain`` agent-loop passthrough.
"""

from __future__ import annotations

import json

ANTHROPIC_DEFAULT_MAX_TOKENS = 4096


def openai_to_anthropic(body: dict, model: str) -> dict:
    """Convert an OpenAI-format request body to Anthropic Messages API format.

    Handles: messages (system extracted), tools, tool_calls, tool_results,
    tool_choice, max_tokens, temperature, model override.
    """
    messages = body.get("messages", [])
    api_messages: list[dict] = []
    system_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        system_parts.append(block)
        else:
            api_msg: dict = {"role": role}
            content = msg.get("content")

            if role == "assistant" and msg.get("tool_calls"):
                blocks: list[dict] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    args_raw = fn.get("arguments", {})
                    if isinstance(args_raw, str):
                        try:
                            tool_input = json.loads(args_raw)
                        except (json.JSONDecodeError, ValueError):
                            tool_input = {}
                    else:
                        tool_input = args_raw or {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": tool_input,
                        }
                    )
                api_msg["content"] = blocks
            elif role == "tool":
                # OpenAI tool result → Anthropic tool_result
                api_msg["role"] = "user"
                api_msg["content"] = [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": content or "",
                    }
                ]
            else:
                api_msg["content"] = content
            api_messages.append(api_msg)

    anthropic_body: dict = {
        "model": model,
        "messages": api_messages,
        "max_tokens": body.get("max_tokens") or ANTHROPIC_DEFAULT_MAX_TOKENS,
    }

    if system_parts:
        anthropic_body["system"] = "\n".join(system_parts)

    if body.get("tools"):
        anthropic_tools: list[dict] = []
        for tool in body["tools"]:
            if tool.get("type") == "function":
                fn = tool.get("function", {})
                anthropic_tools.append(
                    {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
        if anthropic_tools:
            anthropic_body["tools"] = anthropic_tools

    tool_choice = body.get("tool_choice")
    if tool_choice == "auto":
        anthropic_body["tool_choice"] = {"type": "auto"}
    elif tool_choice == "required":
        anthropic_body["tool_choice"] = {"type": "any"}
    elif tool_choice == "none":
        anthropic_body.pop("tools", None)
    elif isinstance(tool_choice, dict) and tool_choice.get("function", {}).get("name"):
        anthropic_body["tool_choice"] = {
            "type": "tool",
            "name": tool_choice["function"]["name"],
        }

    if "temperature" in body:
        anthropic_body["temperature"] = body["temperature"]

    return anthropic_body


def anthropic_to_openai(data: dict, model: str) -> dict:
    """Convert an Anthropic Messages API response to OpenAI chat completion format.

    Handles: text content, tool_use blocks, stop_reason → finish_reason, usage mapping.
    """
    content_blocks = data.get("content", [])
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    tc_index = 0

    for block in content_blocks:
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                    "index": tc_index,
                }
            )
            tc_index += 1

    stop_reason = data.get("stop_reason", "end_turn")
    if stop_reason == "tool_use":
        finish_reason = "tool_calls"
    elif stop_reason == "max_tokens":
        finish_reason = "length"
    else:
        finish_reason = "stop"

    message: dict = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    anthropic_usage = data.get("usage", {})
    prompt_tokens = anthropic_usage.get("input_tokens", 0)
    completion_tokens = anthropic_usage.get("output_tokens", 0)

    return {
        "id": "chatcmpl-" + data.get("id", "unknown"),
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
