"""Tests for SmartSplit — free multi-LLM backend."""

from __future__ import annotations

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


class TestTriageRespond:
    def test_simple_question(self):
        body = {"messages": [{"role": "user", "content": "What are closures?"}]}
        assert triage(body) == TriageDecision.RESPOND

    def test_code_question(self):
        body = {"messages": [{"role": "user", "content": "Write a Python function to sort a list"}]}
        assert triage(body) == TriageDecision.RESPOND

    def test_translation(self):
        body = {"messages": [{"role": "user", "content": "Traduis 'hello' en francais"}]}
        assert triage(body) == TriageDecision.RESPOND

    def test_empty_messages(self):
        body = {"messages": []}
        assert triage(body) == TriageDecision.RESPOND

    def test_no_user_text(self):
        body = {"messages": [{"role": "assistant", "content": "Hi"}]}
        assert triage(body) == TriageDecision.RESPOND


class TestTriageEnrich:
    def test_web_search(self):
        body = {"messages": [{"role": "user", "content": "Search the latest React 19 features released in 2026"}]}
        assert triage(body) == TriageDecision.ENRICH

    def test_factual_with_search(self):
        body = {"messages": [{"role": "user", "content": "What is the current population of Tokyo?"}]}
        assert triage(body) == TriageDecision.ENRICH

    def test_multi_domain(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": "Research OAuth2 best practices and explain how to implement it in Python code",
                }
            ]
        }
        assert triage(body) == TriageDecision.ENRICH


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
