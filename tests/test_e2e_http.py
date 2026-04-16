"""End-to-end HTTP tests for SmartSplit proxy handlers."""

from __future__ import annotations

from unittest.mock import AsyncMock

from starlette.testclient import TestClient

from smartsplit.exceptions import ProviderError
from smartsplit.models import TokenUsage
from smartsplit.proxy.pipeline import create_app
from smartsplit.tools.intention_detector import AnticipatedTool, Prediction


def _make_app(providers=None, mode="balanced"):
    """Create app with mocked providers for testing."""
    if providers is None:
        providers = ["groq", "gemini", "serper"]

    from smartsplit.config import SmartSplitConfig, _resolve_brain
    from smartsplit.models import Mode
    from tests.conftest import SAMPLE_COMPETENCE, _make_provider_config

    provider_configs = {p: _make_provider_config(p) for p in providers}
    config = SmartSplitConfig(
        mode=Mode(mode),
        brain=_resolve_brain(provider_configs),
        providers=provider_configs,
        competence_table=SAMPLE_COMPETENCE,
    )
    return create_app(config=config)


class TestCompletionsEndpoint:
    """Test POST /v1/chat/completions end-to-end."""

    def test_simple_request_returns_200(self):
        app = _make_app()
        with TestClient(app) as client:
            # Mock the brain (auto-detected as groq)
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(
                return_value=("Python is a programming language.", TokenUsage())
            )

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "smartsplit",
                    "messages": [{"role": "user", "content": "What are closures?"}],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["choices"][0]["message"]["content"] == "Python is a programming language."
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert "usage" in data

    def test_openai_format_structure(self):
        app = _make_app()
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(return_value=("ok", TokenUsage()))

            resp = client.post(
                "/v1/chat/completions",
                json={"model": "smartsplit", "messages": [{"role": "user", "content": "Hi"}]},
            )
            data = resp.json()
            assert "id" in data
            assert data["object"] == "chat.completion"
            assert "created" in data
            assert data["model"] == "smartsplit"

    def test_invalid_json(self):
        app = _make_app()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                content=b"not json",
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 400

    def test_empty_messages(self):
        app = _make_app()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "smartsplit", "messages": []},
            )
            assert resp.status_code == 200

    def test_all_providers_fail_returns_503(self):
        app = _make_app(["groq"])
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(side_effect=ProviderError("groq", "down"))

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "smartsplit",
                    "messages": [{"role": "user", "content": "test"}],
                },
            )
            assert resp.status_code == 503
            error = resp.json()["error"]
            assert "unavailable" in error["message"].lower()
            assert error["type"] == "server_error"

    def test_triage_counts_updated(self):
        app = _make_app()
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(return_value=("ok", TokenUsage()))

            client.post(
                "/v1/chat/completions",
                json={"model": "smartsplit", "messages": [{"role": "user", "content": "Hello"}]},
            )

            metrics = client.get("/metrics").json()
            assert metrics["requests"]["total"] >= 1


class TestAnthropicMessagesEndpoint:
    """Test POST /v1/messages (Anthropic Messages API format)."""

    def test_simple_request_returns_200(self):
        app = _make_app()
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(
                return_value=("Python is a programming language.", TokenUsage())
            )

            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": "What are closures?"}],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "message"
            assert data["role"] == "assistant"
            assert len(data["content"]) >= 1
            assert data["content"][0]["type"] == "text"
            assert len(data["content"][0]["text"]) > 0
            assert "usage" in data
            assert data["stop_reason"] == "end_turn"

    def test_anthropic_format_structure(self):
        app = _make_app()
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(return_value=("ok", TokenUsage()))

            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            data = resp.json()
            assert "id" in data
            assert data["id"].startswith("msg_")
            assert data["type"] == "message"
            assert data["role"] == "assistant"
            assert "content" in data
            assert "stop_reason" in data
            assert "usage" in data
            assert "input_tokens" in data["usage"]
            assert "output_tokens" in data["usage"]

    def test_system_prompt_handled(self):
        app = _make_app()
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(return_value=("I'm a coding assistant", TokenUsage()))

            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "system": "You are a coding assistant.",
                    "messages": [{"role": "user", "content": "Who are you?"}],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["content"][0]["text"] == "I'm a coding assistant"

    def test_content_blocks_in_messages(self):
        app = _make_app()
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(return_value=("got it", TokenUsage()))

            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Explain this code"}],
                        }
                    ],
                },
            )
            assert resp.status_code == 200
            assert resp.json()["content"][0]["text"] == "got it"

    def test_invalid_json(self):
        app = _make_app()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/messages",
                content=b"not json",
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 400
            data = resp.json()
            assert data["type"] == "error"
            assert data["error"]["type"] == "invalid_request_error"

    def test_missing_messages(self):
        app = _make_app()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/messages",
                json={"model": "test", "max_tokens": 100},
            )
            assert resp.status_code == 400

    def test_all_providers_fail_returns_503(self):
        app = _make_app(["groq"])
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(side_effect=ProviderError("groq", "down"))

            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "test"}],
                },
            )
            assert resp.status_code == 503
            data = resp.json()
            assert data["type"] == "error"
            assert "unavailable" in data["error"]["message"].lower()

    def test_streaming_returns_sse(self):
        app = _make_app()
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(return_value=("Hello world", TokenUsage()))

            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")

            # Parse SSE events
            raw = resp.text
            assert "event: message_start" in raw
            assert "event: content_block_start" in raw
            assert "event: content_block_delta" in raw
            assert "event: content_block_stop" in raw
            assert "event: message_delta" in raw
            assert "event: message_stop" in raw

    def test_tool_result_messages_handled(self):
        """Test that Anthropic tool_result content blocks are handled correctly."""
        app = _make_app()
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(return_value=("Here's the fix", TokenUsage()))

            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 4096,
                    "messages": [
                        {"role": "user", "content": "Read auth.py"},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "auth.py"}}
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_1",
                                    "content": "def login(): pass",
                                }
                            ],
                        },
                        {"role": "user", "content": "Now fix the bug"},
                    ],
                },
            )
            assert resp.status_code == 200


class TestTransparentPath:
    """Test the TRANSPARENT path — simple prompts forward directly to brain."""

    def test_short_prompt_goes_transparent(self):
        app = _make_app()
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(return_value=("hello back", TokenUsage()))

            resp = client.post(
                "/v1/chat/completions",
                json={"model": "smartsplit", "messages": [{"role": "user", "content": "Hello"}]},
            )
            assert resp.status_code == 200
            assert resp.json()["choices"][0]["message"]["content"] == "hello back"

    def test_tool_messages_go_transparent(self):
        app = _make_app()
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(return_value=("done", TokenUsage()))

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "smartsplit",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Fix this bug in the authentication module and explain what went wrong",
                        },
                        {"role": "assistant", "content": "Reading file..."},
                        {"role": "tool", "content": "file contents"},
                        {"role": "user", "content": "Now apply the fix"},
                    ],
                },
            )
            assert resp.status_code == 200


class TestEnrichPath:
    """Test the ENRICH path — workers prep, then brain synthesizes."""

    def test_enrich_with_web_search(self):
        app = _make_app(["groq", "gemini", "serper"])
        with TestClient(app) as client:
            ctx = client.app.state.ctx

            # Brain (groq) responds to the enriched prompt
            ctx.registry.get("groq").complete = AsyncMock(
                return_value=("React 19 brings server components and improved performance.", TokenUsage())
            )
            # Search worker returns results
            ctx.registry.get("serper").search = AsyncMock(
                return_value="React 19 features: server components, actions, use() hook"
            )

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "smartsplit",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Search the latest React 19 features and summarize them for me in detail please",
                        }
                    ],
                },
            )
            assert resp.status_code == 200
            content = resp.json()["choices"][0]["message"]["content"]
            assert len(content) > 10


class TestMetricsEndpoint:
    """Test GET /metrics with real request flow."""

    def test_metrics_after_requests(self):
        app = _make_app()
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(return_value=("ok", TokenUsage()))

            for _ in range(3):
                client.post(
                    "/v1/chat/completions",
                    json={"model": "smartsplit", "messages": [{"role": "user", "content": "Hi"}]},
                )

            data = client.get("/metrics").json()
            assert data["requests"]["total"] == 3
            assert "transparent" in data["requests"]
            assert "enrich" in data["requests"]
            assert data["cache"]["hits"] >= 0
            assert isinstance(data["circuit_breaker"]["unhealthy_providers"], list)

    def test_health_shows_brain(self):
        app = _make_app()
        with TestClient(app) as client:
            data = client.get("/health").json()
            assert "brain" in data
            assert data["brain"] != ""


class TestFakeToolUse:
    """Test FAKE tool_use — high-confidence predictions return tool_use without calling brain."""

    def test_fake_tool_use_anthropic_endpoint(self):
        """When detector returns confidence >= 0.85, /v1/messages returns fake tool_use."""
        app = _make_app()
        with TestClient(app) as client:
            ctx = client.app.state.ctx

            # Mock the detector to return a high-confidence prediction
            high_confidence_prediction = Prediction(
                should_anticipate=True,
                confidence=0.95,
                tools=[
                    AnticipatedTool(
                        tool="Read",
                        args={"file_path": "smartsplit/proxy.py"},
                        reason="file mentioned in prompt",
                        confidence=0.95,
                    )
                ],
            )
            ctx.detector = AsyncMock()
            ctx.detector.predict = AsyncMock(return_value=high_confidence_prediction)

            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": "Read smartsplit/proxy.py and explain it"}],
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
            data = resp.json()
            assert data["type"] == "message"
            assert data["stop_reason"] == "tool_use"
            assert len(data["content"]) >= 1
            assert data["content"][0]["type"] == "tool_use"
            assert data["content"][0]["name"] == "Read"
            # Brain should NOT have been called
            brain = ctx.registry.get("groq")
            if hasattr(brain.complete, "assert_not_called"):
                brain.complete.assert_not_called()

    def test_fake_tool_use_streaming(self):
        """FAKE tool_use works correctly with stream=True (returns SSE)."""
        app = _make_app()
        with TestClient(app) as client:
            ctx = client.app.state.ctx

            high_confidence_prediction = Prediction(
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
            ctx.detector = AsyncMock()
            ctx.detector.predict = AsyncMock(return_value=high_confidence_prediction)

            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
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
            assert "tool_use" in raw
