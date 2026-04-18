"""Request/response format conversion for OpenAI-compatible API and Anthropic Messages API."""

from __future__ import annotations

import json
import re
import time
import uuid

from pydantic import BaseModel, Field

from smartsplit.models import TokenUsage

# ── OpenAI format ──────────────────────────────────────────────


class OpenAIMessage(BaseModel):
    """A single message in the OpenAI chat format."""

    role: str
    # Content can be a string (text-only) or a list of content parts
    # (multimodal — text + image_url, e.g. GPT-4o vision). Both shapes are
    # forwarded as-is to providers that support them.
    content: str | list[dict[str, object]] | None = None
    # Tool use fields — preserved for agent loop passthrough
    tool_calls: list[dict[str, object]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class OpenAIRequest(BaseModel):
    """Incoming chat completion request in OpenAI format."""

    model: str = "smartsplit"
    messages: list[OpenAIMessage] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    tools: list[dict[str, object]] | None = None
    tool_choice: str | dict[str, object] | None = None


class OpenAIChoice(BaseModel):
    """A single choice in an OpenAI chat completion response."""

    index: int = 0
    message: OpenAIMessage
    finish_reason: str = "stop"


class OpenAIResponse(BaseModel):
    """Outgoing chat completion response in OpenAI format."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "smartsplit"
    choices: list[OpenAIChoice] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)


# ── Helpers ────────────────────────────────────────────────────

_XML_TAG_RE = re.compile(r"<[a-zA-Z][\w-]*>[\s\S]*?</[a-zA-Z][\w-]*>")


def strip_agent_metadata(text: str) -> str:
    """Remove XML-tagged blocks from text, returning only the user's actual input.

    Agents inject metadata into user messages using XML tags (e.g. <system-reminder>,
    <command-name>, <task-notification>). These contain tool descriptions, settings,
    and other context that should NOT influence triage or prediction decisions.

    This is generic to all agent protocols — not specific to any one agent.
    Use for triage/prediction only, NOT for forwarding to the brain.
    """
    cleaned = _XML_TAG_RE.sub("", text).strip()
    # Collapse multiple blank lines left by removed blocks
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned


def extract_prompt(request: OpenAIRequest) -> str:
    """Extract the user's prompt text from an OpenAI-format request.

    Returns the text of the latest user message. For multimodal content (a list
    of parts), concatenates the text parts — non-text parts (images, audio) are
    ignored here because this is used for triage and prediction only.
    """
    for msg in reversed(request.messages):
        if msg.role != "user" or not msg.content:
            continue
        if isinstance(msg.content, str):
            return msg.content
        if isinstance(msg.content, list):
            text_parts = [p.get("text", "") for p in msg.content if isinstance(p, dict) and p.get("type") == "text"]
            joined = " ".join(t for t in text_parts if t)
            if joined:
                return joined
    return ""


def build_response(
    content: str,
    tokens_used: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict:
    """Build an OpenAI-format response dict.

    If prompt_tokens/completion_tokens are provided (from real provider usage),
    they take precedence over the estimated tokens_used.
    """
    if prompt_tokens or completion_tokens:
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    else:
        usage = TokenUsage(
            completion_tokens=tokens_used,
            total_tokens=tokens_used,
        )
    resp = OpenAIResponse(
        choices=[OpenAIChoice(message=OpenAIMessage(role="assistant", content=content))],
        usage=usage,
    )
    return resp.model_dump()


# ── Streaming (SSE) ───────────────────────────────────────────


def stream_chunks(content: str, model: str = "smartsplit") -> list[str]:
    """Build SSE chunks for a streaming response."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    return [
        f"data: {_json_dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n",
        f"data: {_json_dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n",
        f"data: {_json_dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n",
        "data: [DONE]\n\n",
    ]


def _json_dumps(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


# ── Agent-mode SSE conversion ────────────────────────────────


def response_to_sse_chunks(response: dict, model: str = "smartsplit") -> list[str]:
    """Convert a non-streaming OpenAI response to SSE chunks.

    Handles both text responses and tool_call responses.
    """
    chunk_id = response.get("id", "chatcmpl-proxy")
    created = response.get("created", int(time.time()))

    choices = response.get("choices", [])
    if not choices:
        return stream_chunks("", model=model)

    msg = choices[0].get("message", {})
    finish_reason = choices[0].get("finish_reason", "stop")
    content = msg.get("content")
    tool_calls = msg.get("tool_calls")

    chunks: list[str] = []

    # Opening chunk with role
    delta_open: dict = {"role": "assistant"}
    if content:
        delta_open["content"] = ""
    chunks.append(
        "data: "
        + _json_dumps(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta_open, "finish_reason": None}],
            }
        )
        + "\n\n"
    )

    # Content chunk
    if content:
        chunks.append(
            "data: "
            + _json_dumps(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
            )
            + "\n\n"
        )

    # Tool call chunks
    if tool_calls:
        for tc in tool_calls:
            idx = tc.get("index", 0)
            chunks.append(
                "data: "
                + _json_dumps(
                    {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": idx,
                                            "id": tc.get("id", ""),
                                            "type": "function",
                                            "function": {
                                                "name": tc.get("function", {}).get("name", ""),
                                                "arguments": tc.get("function", {}).get("arguments", ""),
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                + "\n\n"
            )

    # Closing chunk
    chunks.append(
        "data: "
        + _json_dumps(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
        )
        + "\n\n"
    )
    chunks.append("data: [DONE]\n\n")

    return chunks


async def iter_sse(chunks: list[str]):
    """Async iterator over SSE chunks."""
    for chunk in chunks:
        yield chunk


# ── Anthropic format ───────────────────────────────────────────


def extract_anthropic_prompt(body: dict) -> str:
    """Extract the last user text from an Anthropic Messages API request.

    Handles content as a plain string or as a list of content blocks.
    Skips tool_result blocks — they are context, not the user's prompt.
    """
    messages = body.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Collect text blocks, ignoring tool_result blocks
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            if text_parts:
                return "\n".join(text_parts)
    return ""


def anthropic_has_tools(body: dict) -> bool:
    """Check whether an Anthropic request includes tool definitions."""
    tools = body.get("tools")
    return isinstance(tools, list) and len(tools) > 0


# Structural thresholds for detecting agent-internal calls (auto-compact, title gen,
# task tracker, etc.). User-initiated agent calls ship a large system prompt
# (CLAUDE.md + tool descriptions, typically >10KB) and leave max_tokens at the
# client default (≥4096). Internal calls use a short dedicated system prompt and
# a capped output. Thresholds are conservative to avoid false positives on short
# legitimate user requests.
_INTERNAL_CALL_SYSTEM_CHARS = 3000
_INTERNAL_CALL_MAX_TOKENS = 1024


def _system_prompt_chars(body: dict) -> int:
    """Return the total character count of the Anthropic ``system`` field."""
    system = body.get("system", "")
    if isinstance(system, list):
        return sum(len(str(b.get("text", ""))) for b in system if isinstance(b, dict))
    return len(str(system))


def is_internal_agent_call(body: dict) -> bool:
    """Heuristic: does this request look like a client-internal background call?

    Agents (Claude Code, Cline, Aider, …) send non-user background requests for
    compaction, title generation, task tracking, suggestions, etc. These share
    a structural fingerprint distinct from user-initiated agent calls:
      - Short dedicated ``system`` prompt (no CLAUDE.md / tool descriptions dump)
      - Low ``max_tokens`` cap (the expected output is a short string)

    When both signals fire, SmartSplit should skip enrichment and prediction
    entirely and forward the request as-is. Enriching an internal call pollutes
    its context and burns free LLM budget on a request the user never made.
    """
    max_tokens = body.get("max_tokens") or 0
    if not (0 < max_tokens < _INTERNAL_CALL_MAX_TOKENS):
        return False
    return _system_prompt_chars(body) < _INTERNAL_CALL_SYSTEM_CHARS


def anthropic_has_tool_named(body: dict, *names: str) -> bool:
    """Check if the agent has any of the named tools available."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return False
    tool_names = {t.get("name", "") for t in tools if isinstance(t, dict)}
    return bool(tool_names & set(names))


def anthropic_has_tool_result(body: dict) -> bool:
    """Check if the last user message contains tool_result blocks."""
    messages = body.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return True
        return False
    return False


def _join_text_blocks(blocks: list) -> str:
    """Concatenate the ``text`` of Anthropic content blocks (tolerate strings and dicts)."""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


def _anthropic_system_to_openai(system: object) -> dict | None:
    """Turn the Anthropic ``system`` field (str or list of blocks) into one OpenAI system message."""
    if isinstance(system, str) and system:
        return {"role": "system", "content": system}
    if isinstance(system, list):
        text = _join_text_blocks(system)
        if text:
            return {"role": "system", "content": text}
    return None


def _anthropic_assistant_to_openai(content: object) -> dict:
    """Convert an Anthropic assistant message (possibly with tool_use blocks) to OpenAI form."""
    if not isinstance(content, list):
        return {"role": "assistant", "content": content if isinstance(content, str) else content}

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                    "index": len(tool_calls),
                }
            )

    msg: dict = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _extract_tool_result_content(tool_content: object) -> str:
    """Flatten the ``content`` of a tool_result block (str or list of text blocks)."""
    if isinstance(tool_content, list):
        return _join_text_blocks(tool_content)
    return tool_content if isinstance(tool_content, str) else ""


def _anthropic_user_to_openai(content: object) -> list[dict]:
    """Convert an Anthropic user message to one or more OpenAI messages (handles tool_result)."""
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if not isinstance(content, list):
        return [{"role": "user", "content": content or ""}]

    has_tool_results = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    if not has_tool_results:
        return [{"role": "user", "content": _join_text_blocks(content)}]

    # Each tool_result becomes its own tool message; text blocks are emitted as a single user msg.
    out: list[dict] = []
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "tool_result":
            out.append(
                {
                    "role": "tool",
                    "content": _extract_tool_result_content(block.get("content", "")),
                    "tool_call_id": block.get("tool_use_id", ""),
                }
            )
        elif btype == "text":
            text_parts.append(block.get("text", ""))
    if text_parts:
        out.append({"role": "user", "content": "\n".join(text_parts)})
    return out


def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool schemas to OpenAI ``function`` tool schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for tool in tools
    ]


def _anthropic_tool_choice_to_openai(tool_choice: object) -> str | dict | None:
    """Map Anthropic ``tool_choice`` to OpenAI's ``auto`` / ``required`` / ``function`` form."""
    if not isinstance(tool_choice, dict):
        return None
    tc_type = tool_choice.get("type", "")
    if tc_type == "auto":
        return "auto"
    if tc_type == "any":
        return "required"
    if tc_type == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    return None


def anthropic_messages_to_openai(body: dict) -> dict:
    """Convert an Anthropic Messages API request body to OpenAI chat completion format.

    This is the reverse of :func:`smartsplit.providers.anthropic_adapter.openai_to_anthropic`.
    """
    openai_messages: list[dict] = []

    system_msg = _anthropic_system_to_openai(body.get("system"))
    if system_msg is not None:
        openai_messages.append(system_msg)

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")
        if role == "assistant":
            openai_messages.append(_anthropic_assistant_to_openai(content))
        elif role == "user":
            openai_messages.extend(_anthropic_user_to_openai(content))
        else:
            openai_messages.append({"role": role, "content": content or ""})

    result: dict = {"model": body.get("model", ""), "messages": openai_messages}
    if body.get("max_tokens"):
        result["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        result["temperature"] = body["temperature"]
    if body.get("stream") is not None:
        result["stream"] = body["stream"]

    if body.get("tools"):
        openai_tools = _anthropic_tools_to_openai(body["tools"])
        if openai_tools:
            result["tools"] = openai_tools

    tool_choice = _anthropic_tool_choice_to_openai(body.get("tool_choice"))
    if tool_choice is not None:
        result["tool_choice"] = tool_choice

    return result


def openai_response_to_anthropic(openai_resp: dict, model: str) -> dict:
    """Convert an OpenAI chat completion response to Anthropic Messages API format."""
    choices = openai_resp.get("choices", [])
    content_blocks: list[dict] = []
    stop_reason = "end_turn"

    if choices:
        msg = choices[0].get("message", {})
        finish_reason = choices[0].get("finish_reason", "stop")

        # Map OpenAI finish_reason → Anthropic stop_reason
        if finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        # Text content
        text = msg.get("content")
        if text:
            content_blocks.append({"type": "text", "text": text})

        # Tool calls → tool_use blocks
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                args_raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except (json.JSONDecodeError, TypeError):
                    args = {}
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", "toolu_" + uuid.uuid4().hex[:24]),
                        "name": fn.get("name", ""),
                        "input": args,
                    }
                )

    # If no content at all, add an empty text block
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    # Map usage
    openai_usage = openai_resp.get("usage", {})
    usage = {
        "input_tokens": openai_usage.get("prompt_tokens", 0),
        "output_tokens": openai_usage.get("completion_tokens", 0),
    }

    return {
        "id": "msg_" + uuid.uuid4().hex[:24],
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def anthropic_response_to_sse(response: dict) -> list[str]:
    """Convert a non-streaming Anthropic response to Anthropic SSE format.

    Produces the full event sequence:
      message_start → content_block_start → content_block_delta → content_block_stop
      → message_delta → message_stop
    """
    chunks: list[str] = []

    # message_start — send the full message envelope (with empty content)
    msg_start = {
        "type": "message_start",
        "message": {
            "id": response.get("id", "msg_unknown"),
            "type": "message",
            "role": "assistant",
            "model": response.get("model", ""),
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": response.get("usage", {}).get("input_tokens", 0), "output_tokens": 0},
        },
    }
    chunks.append("event: message_start\ndata: " + json.dumps(msg_start, ensure_ascii=False) + "\n\n")

    # Content blocks
    content_blocks = response.get("content", [])
    for idx, block in enumerate(content_blocks):
        block_type = block.get("type", "text")

        if block_type == "text":
            # content_block_start
            chunks.append(
                "event: content_block_start\ndata: "
                + json.dumps(
                    {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {"type": "text", "text": ""},
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            # content_block_delta with all text at once (non-streaming)
            text = block.get("text", "")
            if text:
                chunks.append(
                    "event: content_block_delta\ndata: "
                    + json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "text_delta", "text": text},
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
            # content_block_stop
            chunks.append(
                "event: content_block_stop\ndata: "
                + json.dumps({"type": "content_block_stop", "index": idx}, ensure_ascii=False)
                + "\n\n"
            )

        elif block_type == "tool_use":
            # content_block_start
            chunks.append(
                "event: content_block_start\ndata: "
                + json.dumps(
                    {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "input": {},
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            # content_block_delta with the full input JSON
            tool_input = block.get("input", {})
            if tool_input:
                chunks.append(
                    "event: content_block_delta\ndata: "
                    + json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(tool_input, ensure_ascii=False),
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
            # content_block_stop
            chunks.append(
                "event: content_block_stop\ndata: "
                + json.dumps({"type": "content_block_stop", "index": idx}, ensure_ascii=False)
                + "\n\n"
            )

    # message_delta — stop_reason + output usage
    usage = response.get("usage", {})
    chunks.append(
        "event: message_delta\ndata: "
        + json.dumps(
            {
                "type": "message_delta",
                "delta": {"stop_reason": response.get("stop_reason", "end_turn"), "stop_sequence": None},
                "usage": {"output_tokens": usage.get("output_tokens", 0)},
            },
            ensure_ascii=False,
        )
        + "\n\n"
    )

    # message_stop
    chunks.append("event: message_stop\ndata: " + json.dumps({"type": "message_stop"}, ensure_ascii=False) + "\n\n")

    return chunks


def anthropic_to_flat_messages(body: dict) -> list[dict[str, str]]:
    """Convert Anthropic messages to flat role/content dicts for the detector.

    Used by the anticipation pipeline which expects simple message dicts.
    System prompt is included as a system message.
    """
    flat: list[dict[str, str]] = []

    # System prompt
    system = body.get("system")
    if system:
        if isinstance(system, str):
            flat.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text_parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            if text_parts:
                flat.append({"role": "system", "content": "\n".join(text_parts)})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            flat.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_result":
                        # Flatten tool result content
                        tc = block.get("content", "")
                        if isinstance(tc, list):
                            for cb in tc:
                                if isinstance(cb, dict) and cb.get("type") == "text":
                                    text_parts.append(cb.get("text", ""))
                                elif isinstance(cb, str):
                                    text_parts.append(cb)
                        elif isinstance(tc, str):
                            text_parts.append(tc)
                    elif btype == "tool_use":
                        text_parts.append("[tool_use: " + block.get("name", "") + "]")
                elif isinstance(block, str):
                    text_parts.append(block)
            flat.append({"role": role, "content": "\n".join(text_parts)})
        else:
            flat.append({"role": role, "content": content or ""})

    return flat


def inject_anthropic_system_context(body: dict, context_block: str) -> dict:
    """Inject anticipated context into the last user message.

    Appends the context block to the last user message content.
    This preserves Anthropic's prompt cache (system prompt stays cached).
    Returns a modified copy of the body.
    """
    import copy

    new_body = copy.deepcopy(body)
    messages = new_body.get("messages", [])

    # Find the last user message and append context to it
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user":
            content = msg.get("content", "")
            suffix = "\n\n[Additional context from SmartSplit — use if relevant, ignore if not]\n" + context_block
            if isinstance(content, str):
                msg["content"] = content + suffix
            elif isinstance(content, list):
                # Content blocks — append a text block
                msg["content"] = content + [{"type": "text", "text": suffix}]
            break

    return new_body


# ── Fake tool_use responses ─────────────────────────────────────


def build_fake_openai_tool_response(
    tool_calls: list[dict[str, object]],
    model: str = "smartsplit",
) -> dict:
    """Build a fake OpenAI chat completion response with tool_calls.

    Used to anticipate reads: SmartSplit responds with tool_use BEFORE calling
    the brain. The agent executes the reads locally, then sends the results back.
    """
    openai_tool_calls = []
    for i, tc in enumerate(tool_calls):
        tool_name = str(tc.get("tool", ""))
        tool_input = tc.get("input", {})
        openai_tool_calls.append(
            {
                "id": "call_smartsplit_" + uuid.uuid4().hex[:12],
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_input),
                },
                "index": i,
            }
        )

    return {
        "id": "chatcmpl-smartsplit-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": openai_tool_calls,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def build_fake_anthropic_tool_response(
    tool_calls: list[dict[str, object]],
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Build a fake Anthropic Messages API response with tool_use blocks.

    Used to anticipate reads: SmartSplit responds with tool_use BEFORE calling
    the brain. The agent executes the reads locally, then sends the results back.
    """
    content_blocks = []
    for tc in tool_calls:
        content_blocks.append(
            {
                "type": "tool_use",
                "id": "toolu_" + uuid.uuid4().hex[:24],
                "name": str(tc.get("tool", "")),
                "input": tc.get("input", {}),
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
