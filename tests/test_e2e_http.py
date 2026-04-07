"""End-to-end HTTP tests for SmartSplit proxy handlers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

from starlette.testclient import TestClient

from smartsplit.exceptions import ProviderError
from smartsplit.models import TokenUsage
from smartsplit.proxy import create_app


def _make_app(providers=None, mode="balanced"):
    """Create app with mocked providers for testing."""
    if providers is None:
        providers = ["groq", "gemini", "serper"]

    from smartsplit.config import SmartSplitConfig
    from smartsplit.models import Mode
    from tests.conftest import SAMPLE_COMPETENCE, _make_provider_config

    config = SmartSplitConfig(
        mode=Mode(mode),
        providers={p: _make_provider_config(p) for p in providers},
        competence_table=SAMPLE_COMPETENCE,
    )
    return create_app(config=config)


class TestCompletionsEndpoint:
    """Test POST /v1/chat/completions end-to-end."""

    def test_simple_request_returns_200(self):
        app = _make_app()
        with TestClient(app) as client:
            # Mock the provider to return a response
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
            assert "choices" in data
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert len(data["choices"][0]["message"]["content"]) > 0

    def test_streaming_request_returns_sse(self):
        app = _make_app()
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(return_value=("Hello world", TokenUsage()))

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "smartsplit",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            # Validate SSE format
            text = resp.text
            assert "data: " in text
            assert "data: [DONE]" in text

    def test_empty_prompt_returns_response(self):
        app = _make_app()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "smartsplit", "messages": []},
            )
            assert resp.status_code == 200

    def test_malformed_json_returns_error(self):
        app = _make_app()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                content=b"not json",
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 400
            assert "invalid_request_error" in resp.json()["error"]["type"]

    def test_provider_failure_returns_error_message(self):
        app = _make_app(["groq"])
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(side_effect=ProviderError("groq", "API down"))

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "smartsplit",
                    "messages": [{"role": "user", "content": "Hello"}],
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


class TestEnrichPath:
    """Test the ENRICH triage path (web search + respond)."""

    def test_enrich_includes_search_results(self):
        app = _make_app(["groq", "serper"])
        with TestClient(app) as client:
            ctx = client.app.state.ctx
            ctx.registry.get("groq").complete = AsyncMock(
                side_effect=[
                    # decompose returns subtasks
                    (
                        json.dumps(
                            [
                                {"type": "web_search", "content": "React 19 features", "complexity": "low"},
                                {"type": "summarize", "content": "Summarize findings", "complexity": "low"},
                            ]
                        ),
                        TokenUsage(),
                    ),
                    # context injection
                    ("User wants React 19 info.", TokenUsage()),
                    # summarize subtask
                    ("React 19 introduces server components and actions.", TokenUsage()),
                    # synthesis
                    ("React 19 brings server components, actions, and improved performance.", TokenUsage()),
                ]
            )
            ctx.registry.get("serper").search = AsyncMock(
                return_value="React 19 features: server components, actions, use() hook"
            )

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "smartsplit",
                    "messages": [
                        {"role": "user", "content": "Search the latest React 19 features and summarize them for me"}
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

            # Make a few requests
            for _ in range(3):
                client.post(
                    "/v1/chat/completions",
                    json={"model": "smartsplit", "messages": [{"role": "user", "content": "Hi"}]},
                )

            data = client.get("/metrics").json()
            assert data["requests"]["total"] == 3
            assert data["cache"]["hits"] >= 0
            assert isinstance(data["circuit_breaker"]["unhealthy_providers"], list)
