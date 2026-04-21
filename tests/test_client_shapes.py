"""Regression tests for client request shapes — protects against silent data loss.

The proxy must preserve critical fields when forwarding requests to the brain:
- OpenAI format (OpenCode, Cline, Continue, Aider): tool_call_id, tool_calls
- Anthropic format (Claude Code): cache_control, mixed content blocks

Each test below captures a real bug we want to stay dead: a field that went
missing in a past refactor and caused provider 400s or silent cache misses.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from starlette.testclient import TestClient

from smartsplit.models import RouteResult, TaskType, TerminationState
from smartsplit.proxy.formats import inject_anthropic_system_context
from smartsplit.tools.intention_detector import AnticipatedTool, Prediction
from smartsplit.triage.detector import TriageDecision
from tests.test_e2e_http import _make_app


def _fake_enrichment_results() -> list[RouteResult]:
    """One successful web_search RouteResult — enough to trigger the enrichment branch."""
    return [
        RouteResult(
            type=TaskType.WEB_SEARCH,
            response="react 19 migration guide — steps: update react, update react-dom, test",
            provider="serper",
            termination=TerminationState.COMPLETED,
        )
    ]


# ── OpenAI format — tool loop fields ──────────────────────────────


def _openai_tool_loop_body() -> dict:
    """Shape sent by OpenCode / Cline / Continue / Aider after one tool round-trip.

    The `role=tool` message requires `tool_call_id`; the `role=assistant` turn
    that produced it requires `tool_calls`. Providers reject the request with
    HTTP 400 if either is missing.
    """
    return {
        "model": "smartsplit",
        "messages": [
            {
                "role": "user",
                "content": "Search the web for the latest React release notes and summarize them",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"query":"react latest"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "React 19 released with server components.",
            },
            {
                "role": "user",
                "content": "Now search the web for React 19 migration guide and summarize it",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            }
        ],
    }


def _run_agent_enrich(body_payload: dict | None = None) -> dict:
    """Send an OpenAI tool-loop request through the agent+ENRICH path, return the body forwarded to the brain.

    Mocks triage → ENRICH, mocks the enrichment workers, disables the detector so
    no FAKE tool_use shortcut fires, and captures the body passed to
    `proxy_openai_request`.
    """
    payload = body_payload if body_payload is not None else _openai_tool_loop_body()
    app = _make_app(["groq", "gemini", "serper"])
    with TestClient(app) as client:
        ctx = client.app.state.ctx
        ctx.detector = None  # skip FAKE tool_use prediction entirely

        captured: dict = {}

        async def fake_proxy(body: dict) -> dict:
            captured["body"] = body
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

        ctx.registry.get("groq").proxy_openai_request = AsyncMock(side_effect=fake_proxy)

        # Deterministic triage + enrichment — no real LLM/network calls.
        with (
            patch(
                "smartsplit.proxy.pipeline._triage_request",
                AsyncMock(return_value=(TriageDecision.ENRICH, ["web_search"])),
            ),
            patch(
                "smartsplit.proxy.pipeline.enrich_only",
                AsyncMock(return_value=_fake_enrichment_results()),
            ),
        ):
            resp = client.post("/v1/chat/completions", json=payload)

        assert resp.status_code == 200, resp.text
        return captured["body"]


def test_openai_tool_call_id_preserved_in_agent_enrich():
    """Regression: agent-mode + ENRICH must not drop `tool_call_id` from role=tool messages.

    Past bug: body_dict["messages"] was overwritten with a stripped copy that
    kept only role+content, making providers return 400
    ("messages.N.tool_call_id is missing").
    """
    body = _run_agent_enrich()
    tool_msgs = [m for m in body["messages"] if m.get("role") == "tool"]
    assert tool_msgs, "role=tool message should survive the pipeline"
    assert tool_msgs[0]["tool_call_id"] == "call_abc123"


def test_openai_assistant_tool_calls_preserved_in_agent_enrich():
    """Regression: agent-mode + ENRICH must not drop `tool_calls` from role=assistant messages."""
    body = _run_agent_enrich()
    assistant_msgs = [m for m in body["messages"] if m.get("role") == "assistant"]
    assert assistant_msgs, "role=assistant message should survive the pipeline"
    assert assistant_msgs[0].get("tool_calls"), "tool_calls must survive the pipeline"
    assert assistant_msgs[0]["tool_calls"][0]["id"] == "call_abc123"


def test_openai_agent_enrich_injects_context_into_last_user_message():
    """Sanity check — enrichment runs and its context block lands in the last user message."""
    body = _run_agent_enrich()
    last_user = next(m for m in reversed(body["messages"]) if m.get("role") == "user")
    assert "migration guide" in (last_user.get("content") or "")


def test_openai_tool_choice_preserved_in_agent_enrich():
    """Regression: `tool_choice` must survive the pipeline unchanged.

    Some agents (e.g. Continue when forcing a tool) send `tool_choice`; dropping
    it silently changes behavior to `auto`, causing the brain to talk instead
    of tool-calling as the user asked.
    """
    payload = _openai_tool_loop_body()
    payload["tool_choice"] = {"type": "function", "function": {"name": "web_search"}}
    body = _run_agent_enrich(payload)
    assert body.get("tool_choice") == {"type": "function", "function": {"name": "web_search"}}


def test_openai_multimodal_content_list_preserved_in_agent_enrich():
    """Regression: content as a list of parts (multimodal) must not crash or lose parts.

    Past bug: `build_enriched_messages` did ``content + context_block`` assuming
    a string, raising TypeError when content was a list of parts (text + image).
    """
    payload = {
        "model": "smartsplit",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this screenshot? Search the web for context."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            }
        ],
    }
    body = _run_agent_enrich(payload)

    user_content = body["messages"][0]["content"]
    assert isinstance(user_content, list), "multimodal list content should stay a list"
    assert any(part.get("type") == "image_url" for part in user_content), "image part must survive"
    assert any(part.get("type") == "text" for part in user_content), "text part must survive"


def test_openai_multi_turn_tool_loop_all_turns_preserved_in_agent_enrich():
    """Regression: multi-turn tool loops (>1 round) must keep every tool_call_id linked."""
    payload = {
        "model": "smartsplit",
        "messages": [
            {"role": "user", "content": "Find the latest React release and the latest Vue release"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"query":"react latest"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "React 19"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"query":"vue latest"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "Vue 3.5"},
            {"role": "user", "content": "Now summarize both from the web"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            }
        ],
    }
    body = _run_agent_enrich(payload)

    tool_call_ids = {m["tool_call_id"] for m in body["messages"] if m.get("role") == "tool"}
    assert tool_call_ids == {"call_1", "call_2"}, "every tool turn must keep its tool_call_id"

    assistant_ids = [
        tc["id"] for m in body["messages"] if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])
    ]
    assert assistant_ids == ["call_1", "call_2"], "every assistant tool_calls.id must survive"


# ── Anthropic format — content blocks & cache_control ────────────


def test_anthropic_cache_control_preserved_by_context_injection():
    """Regression: enrichment must preserve Anthropic `cache_control` markers.

    Claude Pro / Max users rely on prompt caching (`cache_control: {type: ephemeral}`)
    to keep costs manageable. If SmartSplit rebuilds messages and drops the marker,
    the cache invalidates every request and costs explode silently.
    """
    body = {
        "model": "claude-sonnet-4-6-20250514",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "big stable context", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "fresh question"},
                ],
            }
        ],
    }

    enriched = inject_anthropic_system_context(body, "search result: foo")

    user_blocks = enriched["messages"][0]["content"]
    cached_blocks = [b for b in user_blocks if b.get("cache_control")]
    assert cached_blocks, "cache_control marker must survive context injection"
    assert cached_blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_tool_use_and_tool_result_blocks_preserved_by_context_injection():
    """Regression: context injection on Anthropic body must leave tool_use / tool_result intact.

    Claude Code sends assistant turns with `tool_use` blocks and user turns with
    `tool_result` blocks referencing each other via `id` / `tool_use_id`. An
    earlier refactor flattened these to plain strings, breaking the tool loop
    on the next round-trip.
    """
    body = {
        "model": "claude-sonnet-4-6-20250514",
        "messages": [
            {"role": "user", "content": "Read auth.py and patch it"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me read the file first."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "auth.py"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "def login(): pass"},
                ],
            },
        ],
    }

    enriched = inject_anthropic_system_context(body, "auth best practices: salt hashes")

    # Assistant mixed blocks untouched (tool_use id + name preserved)
    assistant_content = enriched["messages"][1]["content"]
    tool_use_blocks = [b for b in assistant_content if b.get("type") == "tool_use"]
    assert tool_use_blocks and tool_use_blocks[0]["id"] == "toolu_1"
    assert tool_use_blocks[0]["name"] == "Read"

    # tool_result block still linked by tool_use_id — and enrichment text
    # appended as a separate block rather than mangling the result block.
    last_user_content = enriched["messages"][2]["content"]
    tool_result_blocks = [b for b in last_user_content if b.get("type") == "tool_result"]
    assert tool_result_blocks and tool_result_blocks[0]["tool_use_id"] == "toolu_1"
    assert tool_result_blocks[0]["content"] == "def login(): pass"


# ── Streaming SSE — tool_calls / tool_use event shape ─────────────


def _sse_data_chunks(raw: str) -> list[str]:
    """Return the payload of each `data: ...` line in an SSE stream, minus the [DONE] sentinel."""
    payloads: list[str] = []
    for line in raw.splitlines():
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            payloads.append(line[6:])
    return payloads


def test_openai_streaming_fake_tool_use_shape():
    """Regression: FAKE tool_use on `/v1/chat/completions` with stream=True must emit
    OpenAI-format `delta.tool_calls` chunks with id + name + arguments — any agent
    consuming SSE (OpenCode, Cline, Continue, Aider) parses those fields to continue
    the tool loop.
    """
    import json as _json

    app = _make_app(["groq", "gemini", "serper"])
    with TestClient(app) as client:
        ctx = client.app.state.ctx
        ctx.detector = AsyncMock()
        ctx.detector.predict = AsyncMock(
            return_value=Prediction(
                should_anticipate=True,
                confidence=0.95,
                tools=[
                    AnticipatedTool(
                        tool="web_search",
                        args={"query": "latest react"},
                        reason="keyword match",
                        confidence=0.95,
                    )
                ],
            )
        )

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "smartsplit",
                "stream": True,
                "messages": [{"role": "user", "content": "Find the latest React release"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "description": "Search",
                            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                        },
                    }
                ],
            },
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        chunks = [_json.loads(p) for p in _sse_data_chunks(resp.text)]
        tool_call_deltas = [
            tc for c in chunks for tc in (c.get("choices", [{}])[0].get("delta", {}).get("tool_calls", []) or [])
        ]
        assert tool_call_deltas, "streaming must emit at least one tool_calls delta"
        first = tool_call_deltas[0]
        assert first.get("id"), "tool_call id must be present (agent uses it to route tool_result back)"
        assert first.get("function", {}).get("name") == "web_search"
        assert "arguments" in first.get("function", {}), "arguments field must be present (may be empty string)"

        finish_chunks = [c for c in chunks if c.get("choices", [{}])[0].get("finish_reason")]
        assert finish_chunks, "closing chunk with finish_reason must be emitted"
        assert finish_chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_anthropic_streaming_fake_tool_use_shape():
    """Regression: FAKE tool_use on `/v1/messages` with stream=True must emit the
    content_block_start → content_block_stop sequence for a tool_use block with
    id + name populated. Claude Code relies on those to resume the tool loop.
    """
    import json as _json

    app = _make_app(["groq", "gemini", "serper"])
    with TestClient(app) as client:
        ctx = client.app.state.ctx
        ctx.detector = AsyncMock()
        ctx.detector.predict = AsyncMock(
            return_value=Prediction(
                should_anticipate=True,
                confidence=0.95,
                tools=[
                    AnticipatedTool(
                        tool="Read",
                        args={"file_path": "main.py"},
                        reason="file mentioned",
                        confidence=0.95,
                    )
                ],
            )
        )

        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6-20250514",
                "max_tokens": 4096,
                "stream": True,
                "messages": [{"role": "user", "content": "Read main.py"}],
                "tools": [
                    {
                        "name": "Read",
                        "description": "Read a file",
                        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                    }
                ],
            },
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        raw = resp.text
        assert "event: message_start" in raw
        assert "event: content_block_start" in raw
        assert "event: content_block_stop" in raw
        assert "event: message_stop" in raw

        # Parse each content_block_start payload and find the tool_use block
        tool_use_blocks = [
            _json.loads(p).get("content_block", {})
            for p in _sse_data_chunks(raw)
            if '"type": "content_block_start"' in p
        ]
        tool_use = next((b for b in tool_use_blocks if b.get("type") == "tool_use"), None)
        assert tool_use is not None, "a tool_use content_block_start must be present"
        assert tool_use.get("id"), "tool_use id must be populated (links to later tool_result)"
        assert tool_use.get("name") == "Read"
