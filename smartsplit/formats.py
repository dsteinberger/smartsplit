"""Request/response format conversion for OpenAI-compatible API."""

from __future__ import annotations

import json
import time
import uuid

from pydantic import BaseModel, Field

# ── OpenAI format ──────────────────────────────────────────────


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIRequest(BaseModel):
    model: str = "smartsplit"
    messages: list[OpenAIMessage] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChoice(BaseModel):
    index: int = 0
    message: OpenAIMessage
    finish_reason: str = "stop"


class OpenAIResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "smartsplit"
    choices: list[OpenAIChoice] = Field(default_factory=list)
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage)


# ── Helpers ────────────────────────────────────────────────────


def extract_prompt(request: OpenAIRequest) -> str:
    """Extract the user's prompt from an OpenAI-format request."""
    for msg in reversed(request.messages):
        if msg.role == "user":
            return msg.content
    return request.messages[-1].content if request.messages else ""


def build_response(content: str, tokens_used: int = 0) -> dict:
    """Build an OpenAI-format response dict."""
    resp = OpenAIResponse(
        choices=[OpenAIChoice(message=OpenAIMessage(role="assistant", content=content))],
        usage=OpenAIUsage(
            completion_tokens=tokens_used,
            total_tokens=tokens_used,
        ),
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
