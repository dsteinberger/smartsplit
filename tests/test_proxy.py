"""Tests for SmartSplit — free multi-LLM backend."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from smartsplit.formats import (
    OpenAIRequest,
    build_response,
    extract_prompt,
    stream_chunks,
)
from smartsplit.proxy import TriageDecision, create_app, triage

# ── Format tests ───────────────────────────────────────────────


class TestOpenAIFormat:
    def test_extract_prompt(self):
        req = OpenAIRequest(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello world"},
            ]
        )
        assert extract_prompt(req) == "Hello world"

    def test_extract_last_user_message(self):
        req = OpenAIRequest(
            messages=[
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Reply"},
                {"role": "user", "content": "Second"},
            ]
        )
        assert extract_prompt(req) == "Second"

    def test_extract_empty(self):
        assert extract_prompt(OpenAIRequest(messages=[])) == ""

    def test_build_response(self):
        resp = build_response("Hello!", tokens_used=10)
        assert resp["choices"][0]["message"]["content"] == "Hello!"
        assert resp["usage"]["completion_tokens"] == 10

    def test_stream_chunks(self):
        chunks = stream_chunks("Hello!")
        assert len(chunks) == 4
        assert chunks[-1] == "data: [DONE]\n\n"


# ── Triage tests: RESPOND vs ENRICH ───────────────────────────


def _mock_planner(domains: list[str]) -> MagicMock:
    """Create a mock planner that returns given domains from classify_domains."""
    planner = MagicMock()
    planner.classify_domains = AsyncMock(return_value=domains)
    return planner


class TestTriageRespond:
    @pytest.mark.asyncio
    async def test_single_domain(self):
        decision, _ = await triage("Write a Python function", _mock_planner(["code"]))
        assert decision == TriageDecision.RESPOND

    @pytest.mark.asyncio
    async def test_multi_domain_without_web(self):
        decision, _ = await triage("Write code and translate it", _mock_planner(["code", "translation"]))
        assert decision == TriageDecision.RESPOND

    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        planner = _mock_planner([])
        decision, _ = await triage("", planner)
        assert decision == TriageDecision.RESPOND
        planner.classify_domains.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_domains(self):
        decision, _ = await triage("Hi", _mock_planner([]))
        assert decision == TriageDecision.RESPOND


class TestTriageEnrich:
    @pytest.mark.asyncio
    async def test_web_search(self):
        decision, domains = await triage("Search latest news", _mock_planner(["web_search"]))
        assert decision == TriageDecision.ENRICH
        assert "web_search" in domains

    @pytest.mark.asyncio
    async def test_factual(self):
        decision, _ = await triage("What is the population?", _mock_planner(["factual"]))
        assert decision == TriageDecision.ENRICH

    @pytest.mark.asyncio
    async def test_web_search_with_code(self):
        decision, domains = await triage("Search React docs and write code", _mock_planner(["web_search", "code"]))
        assert decision == TriageDecision.ENRICH
        assert "web_search" in domains


# ── App tests ──────────────────────────────────────────────────


class TestAppCreation:
    def test_create_app(self):
        from starlette.applications import Starlette

        app = create_app()
        assert isinstance(app, Starlette)

    def test_routes(self):
        app = create_app()
        paths = [r.path for r in app.routes]
        assert "/v1/chat/completions" in paths
        assert "/health" in paths
        assert "/savings" in paths
        assert "/metrics" in paths


class TestEndpoints:
    def test_health(self, make_config):
        from starlette.testclient import TestClient

        app = create_app(config=make_config(["groq", "gemini"]))
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_metrics_structure(self, make_config):
        from starlette.testclient import TestClient

        app = create_app(config=make_config(["groq", "gemini"]))
        with TestClient(app) as client:
            data = client.get("/metrics").json()
            assert "requests" in data
            assert "enrich" in data["requests"]
            assert "respond" in data["requests"]
            assert "cache" in data
            assert "circuit_breaker" in data
