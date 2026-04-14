"""Interception logic — shared between the lightweight proxy and the mitmproxy addon.

Pure functions that operate on Anthropic request/response dicts.
No dependency on mitmproxy or asyncio — just JSON in, JSON out.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

from smartsplit.tool_registry import (
    DECISIONAL_TOOLS as _DECISIONAL_TOOLS,
)
from smartsplit.tool_registry import (
    DUMB_TOOLS as _DUMB_TOOLS,
)
from smartsplit.tool_registry import (
    FILE_REF_RE as _FILE_REF_RE,
)

logger = logging.getLogger("smartsplit.intercept")

# ── Configuration ──────────────────────────────────────────────

# Tool-Aware Proxy: compression threshold (tokens estimated as chars/4)
_COMPRESS_THRESHOLD_CHARS = 2000  # ~500 tokens — compress above this
_COMPRESS_TARGET_CHARS = 800  # ~200 tokens — target after compression

# Intent keywords for tool prediction
_FIX_KEYWORDS = re.compile(
    r"\b(fix|bug|error|crash|corrige|corriger|erreur|debug|broken|failing)\b",
    re.IGNORECASE,
)
_TEST_KEYWORDS = re.compile(
    r"\b(test|tests|spec|coverage|teste|testes)\b",
    re.IGNORECASE,
)


# ── Helpers ────────────────────────────────────────────────────


def extract_user_prompt(body: dict) -> str:
    """Extract the last user message text from Anthropic format."""
    messages = body.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
                if isinstance(block, str):
                    return block
    return ""


def extract_tool_names(body: dict) -> set[str]:
    """Extract available tool names from the request."""
    tools = body.get("tools", [])
    names = set()
    for tool in tools:
        name = tool.get("name", "")
        if name:
            names.add(name)
    return names


def has_tool_results(body: dict) -> bool:
    """Check if the last message contains tool results (agent executed our faked calls)."""
    messages = body.get("messages", [])
    if not messages:
        return False
    last = messages[-1]
    content = last.get("content")
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    return False


def predict_reads(prompt: str, tool_names: set[str]) -> list[dict]:
    """Predict which read-only tool calls the LLM will make.

    Returns a list of {tool, input} dicts for files to pre-read.
    Uses regex rules — no LLM call, 0ms.
    """
    predictions: list[dict] = []

    # Determine which read tool to use based on what the agent has
    read_tool = "Read" if "Read" in tool_names else "read_file"

    # 1. Files mentioned in prompt → read them
    paths = list(dict.fromkeys(_FILE_REF_RE.findall(prompt)))
    for path in paths[:3]:
        predictions.append(
            {
                "tool": read_tool,
                "input": {"file_path": path} if read_tool == "Read" else {"path": path},
            }
        )

    # 2. Test intent → also read test file
    if _TEST_KEYWORDS.search(prompt) and paths:
        for path in paths[:1]:
            name = Path(path).name
            stem = Path(name).stem
            ext = Path(name).suffix
            if ext == ".py":
                test_path = "test_" + stem + ".py"
                if not any(
                    p.get("input", {}).get("file_path", p.get("input", {}).get("path")) == test_path
                    for p in predictions
                ):
                    predictions.append(
                        {
                            "tool": read_tool,
                            "input": {"file_path": test_path} if read_tool == "Read" else {"path": test_path},
                        }
                    )

    return predictions[:3]


def identify_tool_from_result(messages: list[dict]) -> str:
    """Find the tool name that produced the last tool_result."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return block.get("name", "")
        break
    return ""


def compress_tool_result(content: str, tool_name: str) -> str:
    """Compress a large tool result into a shorter summary."""
    lines = content.split("\n")

    if tool_name in ("Bash", "bash", "execute"):
        if len(lines) > 40:
            head = lines[:20]
            tail = lines[-10:]
            return (
                "\n".join(head)
                + "\n\n[... "
                + str(len(lines) - 30)
                + " lines truncated by SmartSplit ...]\n\n"
                + "\n".join(tail)
            )
        return content

    if tool_name in ("WebSearch", "WebFetch", "web_search", "web_fetch", "fetch"):
        if len(content) > _COMPRESS_TARGET_CHARS:
            return content[:_COMPRESS_TARGET_CHARS] + "\n\n[... truncated by SmartSplit]"
        return content

    if tool_name in ("git_log", "git_diff", "git_blame"):
        if len(lines) > 40:
            head = lines[:30]
            tail = lines[-5:]
            return (
                "\n".join(head)
                + "\n\n[... "
                + str(len(lines) - 35)
                + " lines truncated by SmartSplit ...]\n\n"
                + "\n".join(tail)
            )
        return content

    # Default: truncate to target chars
    if len(content) > _COMPRESS_TARGET_CHARS * 2:
        return content[: _COMPRESS_TARGET_CHARS * 2] + "\n\n[... truncated by SmartSplit]"
    return content


def compress_tool_results_in_body(body: dict) -> tuple[dict, int]:
    """Compress large tool_result blocks in the request body.

    Returns (modified_body, num_compressed).
    Only compresses results from "smart" tools that produce large output.
    """
    messages = body.get("messages", [])
    if not messages:
        return body, 0

    last_tool = identify_tool_from_result(messages)

    if last_tool in _DUMB_TOOLS or last_tool in _DECISIONAL_TOOLS:
        return body, 0

    modified = False
    compressed_count = 0
    new_messages = []

    for msg in messages:
        content = msg.get("content")
        if msg.get("role") == "user" and isinstance(content, list):
            new_blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, str) and len(result_content) > _COMPRESS_THRESHOLD_CHARS:
                        compressed = compress_tool_result(result_content, last_tool)
                        new_block = dict(block)
                        new_block["content"] = compressed
                        new_blocks.append(new_block)
                        modified = True
                        compressed_count += 1
                        logger.info(
                            "Compressed tool_result from %s: %d → %d chars",
                            last_tool,
                            len(result_content),
                            len(compressed),
                        )
                    elif isinstance(result_content, list):
                        new_sub_blocks = []
                        for sub in result_content:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                text = sub.get("text", "")
                                if len(text) > _COMPRESS_THRESHOLD_CHARS:
                                    compressed = compress_tool_result(text, last_tool)
                                    new_sub_blocks.append({"type": "text", "text": compressed})
                                    modified = True
                                    compressed_count += 1
                                    logger.info(
                                        "Compressed text block from %s: %d → %d chars",
                                        last_tool,
                                        len(text),
                                        len(compressed),
                                    )
                                else:
                                    new_sub_blocks.append(sub)
                            else:
                                new_sub_blocks.append(sub)
                        new_block = dict(block)
                        new_block["content"] = new_sub_blocks
                        new_blocks.append(new_block)
                    else:
                        new_blocks.append(block)
                else:
                    new_blocks.append(block)
            new_msg = dict(msg)
            new_msg["content"] = new_blocks
            new_messages.append(new_msg)
        else:
            new_messages.append(msg)

    if not modified:
        return body, 0

    new_body = dict(body)
    new_body["messages"] = new_messages
    return new_body, compressed_count


def build_fake_response(tool_calls: list[dict], model: str = "claude-sonnet-4-20250514") -> dict:
    """Build a fake Anthropic response with tool_use blocks."""
    content_blocks = []
    for tc in tool_calls:
        content_blocks.append(
            {
                "type": "tool_use",
                "id": "toolu_" + uuid.uuid4().hex[:24],
                "name": tc["tool"],
                "input": tc["input"],
            }
        )

    return {
        "id": "msg_smartsplit_" + uuid.uuid4().hex[:12],
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def build_sse_response(response_body: dict) -> bytes:
    """Convert a non-streaming response to Anthropic SSE format for streaming clients."""
    events = []

    msg_start = dict(response_body)
    msg_start["content"] = []
    events.append("event: message_start\ndata: " + json.dumps({"type": "message_start", "message": msg_start}) + "\n\n")

    for i, block in enumerate(response_body.get("content", [])):
        events.append(
            "event: content_block_start\n"
            "data: " + json.dumps({"type": "content_block_start", "index": i, "content_block": block}) + "\n\n"
        )
        events.append(
            "event: content_block_stop\ndata: " + json.dumps({"type": "content_block_stop", "index": i}) + "\n\n"
        )

    events.append(
        "event: message_delta\n"
        "data: "
        + json.dumps(
            {
                "type": "message_delta",
                "delta": {"stop_reason": response_body.get("stop_reason", "tool_use"), "stop_sequence": None},
                "usage": {"output_tokens": 0},
            }
        )
        + "\n\n"
    )

    events.append("event: message_stop\ndata: " + json.dumps({"type": "message_stop"}) + "\n\n")

    return "".join(events).encode("utf-8")
