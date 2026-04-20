"""Tests for SmartSplit — multi-LLM backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from smartsplit.proxy.formats import (
    OpenAIMessage,
    OpenAIRequest,
    anthropic_has_tools,
    anthropic_messages_to_openai,
    anthropic_response_to_sse,
    anthropic_to_flat_messages,
    build_response,
    extract_anthropic_prompt,
    extract_prompt,
    inject_anthropic_system_context,
    openai_response_to_anthropic,
    response_to_sse_chunks,
    stream_chunks,
)
from smartsplit.proxy.pipeline import create_app
from smartsplit.tools.anticipation import (
    extract_actual_tool_calls,
    extract_already_read_paths,
    extract_project_context,
    extract_recently_written_paths,
    inject_anticipated_context,
)
from smartsplit.triage.detector import TriageDecision, detect

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


# ── Detector tests: TRANSPARENT vs ENRICH ────────────────────


@dataclass
class DetectorCase:
    """A single detector test case."""

    prompt: str
    expected: str  # "transparent" or "enrich"
    expected_types: list[str] | None = None  # enrichment types to check (None = don't check)
    messages: list[dict[str, str]] | None = None
    id: str = ""


# ── TRANSPARENT cases — should pass straight to brain ─────────

_TRANSPARENT_CASES = [
    DetectorCase(prompt="", expected="transparent", id="empty"),
    DetectorCase(prompt="Hello", expected="transparent", id="greeting"),
    DetectorCase(prompt="Fix the bug in line 42", expected="transparent", id="short_code"),
    DetectorCase(prompt="What is a binary tree?", expected="transparent", id="simple_question"),
    DetectorCase(prompt="Write a function", expected="transparent", id="short_prompt"),
    DetectorCase(
        prompt="Write a Python function to add two numbers and return the result",
        expected="transparent",
        id="simple_code",
    ),
    DetectorCase(prompt="Explain how async/await works in Python", expected="transparent", id="explanation"),
    DetectorCase(prompt="Convert this to TypeScript", expected="transparent", id="conversion"),
    DetectorCase(prompt="Add error handling to this function", expected="transparent", id="simple_task"),
    DetectorCase(prompt="What does this regex do: ^[a-z]+$", expected="transparent", id="regex_question"),
    DetectorCase(prompt="List all Python string methods", expected="transparent", id="factual_short"),
    DetectorCase(prompt="Write unit tests for the User class", expected="transparent", id="test_request"),
    # Tool calls — agentic flow
    DetectorCase(
        prompt="Apply the fix",
        messages=[
            {"role": "user", "content": "Fix the auth bug"},
            {"role": "assistant", "content": "Reading..."},
            {"role": "tool", "content": "def login(): pass"},
        ],
        expected="transparent",
        id="tool_messages",
    ),
    DetectorCase(
        prompt="Call the API",
        messages=[
            {"role": "user", "content": "Call the API"},
            {"role": "function", "content": '{"result": "ok"}'},
        ],
        expected="transparent",
        id="function_messages",
    ),
    # Long prompt but single domain — not worth enriching
    DetectorCase(
        prompt="Write a Python class that implements a thread-safe queue with put and get methods using locks",
        expected="transparent",
        id="long_single_domain",
    ),
    # Below _ENRICH_MIN_CHARS (50) — stays transparent even with analysis keyword
    DetectorCase(
        prompt="Refactor this",
        expected="transparent",
        id="short_analysis_keyword",
    ),
    # Just above _ENRICH_MIN_CHARS but short of _PRE_ANALYSIS_MIN_CHARS (120) and no compare keyword
    DetectorCase(
        prompt="Can you help me write a function that squares numbers?",
        expected="transparent",
        id="medium_no_trigger",
    ),
    # Short web_search prompt (< 50 chars) stays ENRICH because web_search is detected
    # BEFORE the length filter. Kept in ENRICH_CASES below, not here — this is just a doc note.
    # Large system message (IDE-injected file) should NOT trigger context_summary
    DetectorCase(
        prompt="What do you think?",
        messages=[
            {"role": "system", "content": "x" * 10000},
            {"role": "user", "content": "What do you think?"},
        ],
        expected="transparent",
        id="large_system_message",
    ),
]

# ── ENRICH cases — should trigger worker enrichment ───────────

_ENRICH_CASES = [
    # Web search — even short prompts
    DetectorCase(
        prompt="Who won the latest F1 race in 2026?", expected="enrich", expected_types=["web_search"], id="web_short"
    ),
    # Regression guard: short web_search prompt bypasses the length filter (< _ENRICH_MIN_CHARS)
    DetectorCase(
        prompt="latest AI news 2026",
        expected="enrich",
        expected_types=["web_search"],
        id="web_very_short_bypasses_length",
    ),
    # New 130-char pre_analysis case — previously TRANSPARENT (under old 200 threshold),
    # now ENRICH since we lowered _PRE_ANALYSIS_MIN_CHARS to 120.
    DetectorCase(
        prompt="Please refactor this payment function to handle currency rounding, null amounts, and concurrent update edge cases safely.",
        expected="enrich",
        expected_types=["pre_analysis"],
        id="pre_analysis_medium_length",
    ),
    DetectorCase(
        prompt="What happened in tech news today?", expected="enrich", expected_types=["web_search"], id="web_today"
    ),
    DetectorCase(
        prompt="Search for the current Python version",
        expected="enrich",
        expected_types=["web_search"],
        id="web_search_keyword",
    ),
    DetectorCase(
        prompt="What are the latest features in Python 3.13? Search for the most current information.",
        expected="enrich",
        expected_types=["web_search"],
        id="web_long",
    ),
    DetectorCase(
        prompt="What are the recent changes in React 19?",
        expected="enrich",
        expected_types=["web_search"],
        id="web_recent",
    ),
    # Comparisons
    DetectorCase(
        prompt="Compare Redis vs Memcached for session caching in a production environment with high traffic.",
        expected="enrich",
        expected_types=["multi_perspective"],
        id="compare_vs",
    ),
    DetectorCase(
        prompt="What are the pros and cons of microservices versus monolithic architecture for a small team?",
        expected="enrich",
        expected_types=["multi_perspective"],
        id="compare_pros_cons",
    ),
    DetectorCase(
        prompt="Which is better for a REST API: Flask or FastAPI? Consider performance and ease of use.",
        expected="enrich",
        expected_types=["multi_perspective"],
        id="compare_which_better",
    ),
    # Analysis / review (long prompts)
    DetectorCase(
        prompt="Refactor this entire authentication module to use JWT tokens instead of session cookies. " * 3,
        expected="enrich",
        expected_types=["pre_analysis"],
        id="refactor",
    ),
    DetectorCase(
        prompt="Review this code and identify all potential security vulnerabilities and performance issues. " * 3,
        expected="enrich",
        expected_types=["pre_analysis"],
        id="review",
    ),
    DetectorCase(
        prompt="Analyze this database schema and suggest optimizations for query performance at scale. " * 3,
        expected="enrich",
        expected_types=["pre_analysis"],
        id="analyze",
    ),
    DetectorCase(
        prompt="Audit this API for OWASP top 10 vulnerabilities and provide a detailed report with fixes. " * 3,
        expected="enrich",
        expected_types=["pre_analysis"],
        id="audit",
    ),
    # Long conversation history
    DetectorCase(
        prompt="Continue",
        messages=[{"role": "user", "content": f"Message {i} content"} for i in range(15)],
        expected="enrich",
        expected_types=["context_summary"],
        id="long_history_messages",
    ),
    DetectorCase(
        prompt="What next?",
        messages=[{"role": "user", "content": "x" * 1000} for _ in range(6)],
        expected="enrich",
        expected_types=["context_summary"],
        id="long_history_chars",
    ),
    # Multi-domain (long prompts with 2+ domains)
    DetectorCase(
        prompt=(
            "Write a Python function that validates email addresses using regex, "
            "and translate the docstring to French. Also explain the regex pattern "
            "and its edge cases in detail for documentation. "
            "Make sure the function handles international domain names properly."
        ),
        expected="enrich",
        id="multi_domain",
    ),
    # Multiple enrichment types
    DetectorCase(
        prompt=(
            "Search the latest news about Python 3.13 and compare it vs Python 3.12. "
            "Analyze the performance improvements in detail and explain the tradeoffs. "
        )
        * 2,
        expected="enrich",
        id="multiple_types",
    ),
    # ── Multilingual cases ───────────────────────────────────
    # Français — comparaison
    DetectorCase(
        prompt="Comparer les avantages et inconvénients de Redis versus Memcached pour notre système de cache distribué",
        expected="enrich",
        expected_types=["multi_perspective"],
        id="compare_fr",
    ),
    # Français — analyse (long)
    DetectorCase(
        prompt="Refactoriser le module d'authentification et améliorer la gestion des erreurs dans tout le projet. "
        * 3,
        expected="enrich",
        expected_types=["pre_analysis"],
        id="analyze_fr",
    ),
    # Español — comparaison
    DetectorCase(
        prompt="Comparar las ventajas y desventajas de PostgreSQL versus MongoDB para nuestro proyecto de backend",
        expected="enrich",
        expected_types=["multi_perspective"],
        id="compare_es",
    ),
    # Deutsch — analyse (long)
    DetectorCase(
        prompt="Analysieren und verbessern Sie die Fehlerbehandlung im gesamten Authentifizierungsmodul des Projekts. "
        * 3,
        expected="enrich",
        expected_types=["pre_analysis"],
        id="analyze_de",
    ),
    # 中文 — web search
    DetectorCase(
        prompt="搜索最新的React框架更新和最佳实践",
        expected="enrich",
        expected_types=["web_search"],
        id="web_zh",
    ),
    # 日本語 — analyse (long, must be >200 chars for pre_analysis)
    DetectorCase(
        prompt="このプロジェクトの認証モジュールをリファクタリングして、エラー処理を改善してください。コード全体を見直してください。セキュリティの脆弱性も確認してください。パフォーマンスの問題も分析してください。"
        * 3,
        expected="enrich",
        expected_types=["pre_analysis"],
        id="analyze_ja",
    ),
    # 한국어 — comparaison (must be >80 chars for enrichment)
    DetectorCase(
        prompt="PostgreSQL과 MongoDB의 장단점을 비교하고 우리 프로젝트에 가장 적합한 데이터베이스를 추천해주세요. 성능, 확장성, 유지보수성을 고려해주세요.",
        expected="enrich",
        expected_types=["multi_perspective"],
        id="compare_ko",
    ),
    # Русский — analyse (long)
    DetectorCase(
        prompt="Проанализировать и улучшить обработку ошибок в модуле аутентификации всего проекта. " * 3,
        expected="enrich",
        expected_types=["pre_analysis"],
        id="analyze_ru",
    ),
    # Português — web search
    DetectorCase(
        prompt="Pesquisar as últimas notícias sobre inteligência artificial",
        expected="enrich",
        expected_types=["web_search"],
        id="web_pt",
    ),
]


class TestDetectorTransparent:
    @pytest.mark.parametrize(
        "case",
        _TRANSPARENT_CASES,
        ids=[c.id for c in _TRANSPARENT_CASES],
    )
    def test_transparent(self, case: DetectorCase):
        decision, _ = detect(case.prompt, case.messages)
        assert decision == TriageDecision.TRANSPARENT, f"Expected TRANSPARENT for: {case.prompt[:60]!r}"


class TestDetectorEnrich:
    @pytest.mark.parametrize(
        "case",
        _ENRICH_CASES,
        ids=[c.id for c in _ENRICH_CASES],
    )
    def test_enrich(self, case: DetectorCase):
        decision, types = detect(case.prompt, case.messages)
        assert decision == TriageDecision.ENRICH, f"Expected ENRICH for: {case.prompt[:60]!r}"
        if case.expected_types:
            for expected_type in case.expected_types:
                assert expected_type in types, f"Expected {expected_type} in {types}"
        if case.id == "multiple_types":
            assert len(types) >= 2, f"Expected 2+ enrichment types, got {types}"


class TestDetectorProxyMode:
    """proxy_mode=True skips context_summary since the brain already has full history."""

    def test_proxy_mode_skips_context_summary_on_message_count(self):
        messages = [{"role": "user", "content": f"Message {i} content"} for i in range(15)]
        # Default (non-proxy): triggers context_summary
        _, default_types = detect("Continue", messages)
        assert "context_summary" in default_types
        # Proxy mode: context_summary skipped
        decision, proxy_types = detect("Continue", messages, proxy_mode=True)
        assert "context_summary" not in proxy_types
        assert decision == TriageDecision.TRANSPARENT

    def test_proxy_mode_skips_context_summary_on_char_count(self):
        messages = [{"role": "user", "content": "x" * 1000} for _ in range(6)]
        _, default_types = detect("What next?", messages)
        assert "context_summary" in default_types
        decision, proxy_types = detect("What next?", messages, proxy_mode=True)
        assert "context_summary" not in proxy_types
        assert decision == TriageDecision.TRANSPARENT

    def test_proxy_mode_still_detects_web_search(self):
        # Long history + web_search keyword → web_search still fires in proxy mode
        messages = [{"role": "user", "content": "x" * 1000} for _ in range(6)]
        decision, types = detect("latest AI news 2026", messages, proxy_mode=True)
        assert "web_search" in types
        assert "context_summary" not in types
        assert decision == TriageDecision.ENRICH


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
        assert "/v1/messages" in paths
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
            assert "brain" in resp.json()

    def test_metrics_structure(self, make_config):
        from starlette.testclient import TestClient

        app = create_app(config=make_config(["groq", "gemini"]))
        with TestClient(app) as client:
            data = client.get("/metrics").json()
            assert "requests" in data
            assert "transparent" in data["requests"]
            assert "enrich" in data["requests"]
            assert "cache" in data
            assert "circuit_breaker" in data


# ── Agent mode tests ─────────────────────────────────────────


class TestAgentMode:
    # ── extract_already_read_paths ──────────────────────────

    def testextract_already_read_paths(self):
        messages = [
            {"role": "user", "content": "Read auth.py"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": '{"path": "auth.py"}'}},
                    {"function": {"name": "Read", "arguments": '{"file_path": "router.py"}'}},
                ],
            },
            {"role": "tool", "content": "def login(): pass"},
        ]
        paths = extract_already_read_paths(messages)
        assert "auth.py" in paths
        assert "router.py" in paths
        assert len(paths) == 2

    def testextract_already_read_paths_empty(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        paths = extract_already_read_paths(messages)
        assert paths == set()

    # ── extract_recently_written_paths ──────────────────────

    def testextract_recently_written_paths(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "write_file", "arguments": '{"path": "new.py"}'}},
                    {"function": {"name": "Edit", "arguments": '{"file_path": "old.py"}'}},
                ],
            },
            {"role": "tool", "content": "ok"},
        ]
        paths = extract_recently_written_paths(messages)
        assert "new.py" in paths
        assert "old.py" in paths

    def testextract_recently_written_paths_ignores_reads(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": '{"path": "src/main.py"}'}},
                ],
            },
            {"role": "tool", "content": "code here"},
        ]
        paths = extract_recently_written_paths(messages)
        assert paths == set()

    # ── inject_anticipated_context ──────────────────────────

    def testinject_anticipated_context_adds_system_message(self):
        original = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Read my code"},
        ]
        anticipated = [
            {"tool": "read_file", "args_summary": "path=auth.py", "content": "def login(): pass"},
        ]
        result = inject_anticipated_context(original, anticipated)
        # Injected system message at position 0
        assert result[0]["role"] == "system"
        assert "[SmartSplit" in result[0]["content"]
        assert "read_file" in result[0]["content"]
        assert "def login(): pass" in result[0]["content"]
        # Original messages preserved after the injected one
        assert len(result) == len(original) + 1
        assert result[1] == original[0]
        assert result[2] == original[1]

    def testinject_anticipated_context_empty(self):
        original = [
            {"role": "user", "content": "Hello"},
        ]
        result = inject_anticipated_context(original, [])
        assert result is original  # same reference, no copy

    # ── extract_project_context ─────────────────────────────

    def testextract_project_context_from_system(self):
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant for this Python project."},
            {"role": "user", "content": "Help me"},
        ]
        ctx = extract_project_context(messages)
        assert "PROJECT CONTEXT" in ctx
        assert "You are a helpful coding assistant" in ctx
        # Truncated to 500 chars max
        assert len(ctx) < 600

    def testextract_project_context_no_system(self):
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        ctx = extract_project_context(messages)
        assert ctx == ""

    # ── response_to_sse_chunks (text) ───────────────────────

    def testresponse_to_sse_chunks_text(self):
        response = {
            "id": "chatcmpl-test123",
            "created": 1700000000,
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello world"},
                    "finish_reason": "stop",
                }
            ],
        }
        chunks = response_to_sse_chunks(response, model="test-model")

        # All chunks start with "data: "
        for chunk in chunks:
            assert chunk.startswith("data: ")

        # Last chunk is [DONE]
        assert chunks[-1] == "data: [DONE]\n\n"

        # Content is in one of the middle chunks
        content_found = False
        for chunk in chunks[:-1]:
            payload = chunk.removeprefix("data: ").strip()
            parsed = json.loads(payload)
            delta = parsed["choices"][0]["delta"]
            if delta.get("content") == "Hello world":
                content_found = True
        assert content_found

        # All non-DONE chunks contain valid JSON
        for chunk in chunks[:-1]:
            payload = chunk.removeprefix("data: ").strip()
            json.loads(payload)  # should not raise

    # ── response_to_sse_chunks (tool_calls) ─────────────────

    def testresponse_to_sse_chunks_tool_calls(self):
        response = {
            "id": "chatcmpl-tools",
            "created": 1700000000,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "index": 0,
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "main.py"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        chunks = response_to_sse_chunks(response)

        # Last chunk is [DONE]
        assert chunks[-1] == "data: [DONE]\n\n"

        # Find the tool_calls chunk
        tool_chunk_found = False
        for chunk in chunks[:-1]:
            payload = chunk.removeprefix("data: ").strip()
            parsed = json.loads(payload)
            delta = parsed["choices"][0]["delta"]
            if "tool_calls" in delta:
                tc = delta["tool_calls"][0]
                assert tc["id"] == "call_abc"
                assert tc["type"] == "function"
                assert tc["function"]["name"] == "read_file"
                assert tc["function"]["arguments"] == '{"path": "main.py"}'
                tool_chunk_found = True
        assert tool_chunk_found

    # ── extract_actual_tool_calls ───────────────────────────

    def testextract_actual_tool_calls(self):
        messages = [
            {"role": "user", "content": "Read the files"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
                    {"function": {"name": "grep", "arguments": '{"pattern": "TODO"}'}},
                ],
            },
            {"role": "tool", "content": "file content"},
        ]
        calls = extract_actual_tool_calls(messages)
        assert len(calls) == 2
        assert calls[0] == {"tool": "read_file", "args": {"path": "a.py"}}
        assert calls[1] == {"tool": "grep", "args": {"pattern": "TODO"}}

    # ── OpenAIMessage with tool fields ───────────────────────

    def test_openai_message_with_tool_fields(self):
        # Assistant message with tool_calls and no content
        msg = OpenAIMessage(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "x.py"}'},
                }
            ],
        )
        assert msg.role == "assistant"
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["function"]["name"] == "read_file"

        # Tool result message
        tool_msg = OpenAIMessage(
            role="tool",
            content="file contents here",
            tool_call_id="call_1",
        )
        assert tool_msg.role == "tool"
        assert tool_msg.tool_call_id == "call_1"
        assert tool_msg.content == "file contents here"


# ── Anthropic format tests ─────────────────────────────────────


class TestAnthropicFormatHelpers:
    """Test Anthropic Messages API format conversion helpers."""

    # ── extract_anthropic_prompt ────────────────────────────

    def test_extract_prompt_simple_string(self):
        body = {"messages": [{"role": "user", "content": "Hello world"}]}
        assert extract_anthropic_prompt(body) == "Hello world"

    def test_extract_prompt_content_blocks(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Explain this code"}],
                }
            ]
        }
        assert extract_anthropic_prompt(body) == "Explain this code"

    def test_extract_prompt_skips_tool_result(self):
        body = {
            "messages": [
                {"role": "user", "content": "Read auth.py"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "auth.py"}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file contents"}],
                },
            ]
        }
        # tool_result-only message has no text blocks → falls back to previous user message
        prompt = extract_anthropic_prompt(body)
        assert prompt == "Read auth.py"

    def test_extract_prompt_last_user_message(self):
        body = {
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Answer"},
                {"role": "user", "content": "Follow up"},
            ]
        }
        assert extract_anthropic_prompt(body) == "Follow up"

    def test_extract_prompt_empty(self):
        assert extract_anthropic_prompt({"messages": []}) == ""

    # ── anthropic_has_tools ────────────────────────────────

    def test_has_tools_true(self):
        body = {
            "tools": [{"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        assert anthropic_has_tools(body) is True

    def test_has_tools_false(self):
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        assert anthropic_has_tools(body) is False

    def test_has_tools_empty_list(self):
        body = {"tools": [], "messages": [{"role": "user", "content": "Hi"}]}
        assert anthropic_has_tools(body) is False

    # ── anthropic_messages_to_openai ────────────────────────

    def test_convert_simple_messages(self):
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_messages_to_openai(body)
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["max_tokens"] == 4096
        assert len(result["messages"]) == 1
        assert result["messages"][0] == {"role": "user", "content": "Hello"}

    def test_convert_system_prompt(self):
        body = {
            "model": "test",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_messages_to_openai(body)
        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert result["messages"][1] == {"role": "user", "content": "Hi"}

    def test_convert_tool_use_messages(self):
        body = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "Read auth.py"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "auth.py"}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "def login(): pass"}],
                },
            ],
        }
        result = anthropic_messages_to_openai(body)
        messages = result["messages"]

        # User message
        assert messages[0] == {"role": "user", "content": "Read auth.py"}

        # Assistant with tool_calls
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] is None
        assert len(messages[1]["tool_calls"]) == 1
        tc = messages[1]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "Read"
        assert json.loads(tc["function"]["arguments"]) == {"path": "auth.py"}

        # Tool result
        assert messages[2]["role"] == "tool"
        assert messages[2]["content"] == "def login(): pass"
        assert messages[2]["tool_call_id"] == "toolu_1"

    def test_convert_tools(self):
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }
        result = anthropic_messages_to_openai(body)
        assert "tools" in result
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "Read"
        assert tool["function"]["description"] == "Read a file"

    def test_convert_tool_choice(self):
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "auto"},
        }
        result = anthropic_messages_to_openai(body)
        assert result["tool_choice"] == "auto"

    def test_convert_tool_choice_any(self):
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "any"},
        }
        result = anthropic_messages_to_openai(body)
        assert result["tool_choice"] == "required"

    # ── openai_response_to_anthropic ────────────────────────

    def test_convert_text_response(self):
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = openai_response_to_anthropic(openai_resp, "claude-sonnet-4-20250514")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_convert_tool_calls_response(self):
        openai_resp = {
            "id": "chatcmpl-456",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "Read", "arguments": '{"path": "auth.py"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        result = openai_response_to_anthropic(openai_resp, "test-model")
        assert result["stop_reason"] == "tool_use"
        # Should have a tool_use block
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "Read"
        assert tool_blocks[0]["input"] == {"path": "auth.py"}
        assert tool_blocks[0]["id"] == "call_abc"

    def test_convert_length_stop_reason(self):
        openai_resp = {
            "id": "chatcmpl-789",
            "choices": [{"message": {"role": "assistant", "content": "Truncated..."}, "finish_reason": "length"}],
            "usage": {},
        }
        result = openai_response_to_anthropic(openai_resp, "test")
        assert result["stop_reason"] == "max_tokens"

    # ── anthropic_response_to_sse ────────────────────────────

    def test_text_response_sse(self):
        response = {
            "id": "msg_test123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hello world"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        chunks = anthropic_response_to_sse(response)

        # Check event types present
        events = [c.split("\n")[0] for c in chunks]
        assert "event: message_start" in events
        assert "event: content_block_start" in events
        assert "event: content_block_delta" in events
        assert "event: content_block_stop" in events
        assert "event: message_delta" in events
        assert "event: message_stop" in events

        # Check the text content is in a delta
        text_found = False
        for chunk in chunks:
            if "text_delta" in chunk:
                data_line = [line for line in chunk.split("\n") if line.startswith("data: ")][0]
                data = json.loads(data_line[6:])
                if data.get("delta", {}).get("text") == "Hello world":
                    text_found = True
        assert text_found

    def test_tool_use_response_sse(self):
        response = {
            "id": "msg_test456",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [{"type": "tool_use", "id": "toolu_abc", "name": "Read", "input": {"path": "auth.py"}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        chunks = anthropic_response_to_sse(response)

        # Should have tool_use content_block_start
        tool_start_found = False
        for chunk in chunks:
            if "content_block_start" in chunk and "tool_use" in chunk:
                data_line = [line for line in chunk.split("\n") if line.startswith("data: ")][0]
                data = json.loads(data_line[6:])
                block = data.get("content_block", {})
                if block.get("type") == "tool_use" and block.get("name") == "Read":
                    tool_start_found = True
        assert tool_start_found

    # ── anthropic_to_flat_messages ─────────────────────────

    def test_flat_messages_simple(self):
        body = {
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        flat = anthropic_to_flat_messages(body)
        assert len(flat) == 3
        assert flat[0] == {"role": "system", "content": "You are helpful."}
        assert flat[1] == {"role": "user", "content": "Hello"}
        assert flat[2] == {"role": "assistant", "content": "Hi there"}

    def test_flat_messages_with_content_blocks(self):
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ]
        }
        flat = anthropic_to_flat_messages(body)
        assert flat[0]["content"] == "Hello"

    def test_flat_messages_tool_result(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file contents"}],
                }
            ]
        }
        flat = anthropic_to_flat_messages(body)
        assert "file contents" in flat[0]["content"]

    # ── inject_anthropic_system_context ──────────────────────

    def test_inject_into_last_user_message(self):
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        result = inject_anthropic_system_context(body, "CONTEXT HERE")
        assert "CONTEXT HERE" in result["messages"][-1]["content"]
        assert "Hi" in result["messages"][-1]["content"]
        # Original unchanged
        assert body["messages"][0]["content"] == "Hi"

    def test_inject_preserves_system_prompt(self):
        body = {"system": "You are helpful.", "messages": [{"role": "user", "content": "Hi"}]}
        result = inject_anthropic_system_context(body, "CONTEXT")
        # System prompt untouched (preserves cache)
        assert result["system"] == "You are helpful."
        # Context in last user message
        assert "CONTEXT" in result["messages"][-1]["content"]

    def test_inject_into_content_blocks(self):
        body = {
            "system": [{"type": "text", "text": "You are helpful."}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        }
        result = inject_anthropic_system_context(body, "CONTEXT")
        # System untouched
        assert result["system"][0]["text"] == "You are helpful."
        # Context appended as new block in last user message
        last_msg = result["messages"][-1]["content"]
        assert isinstance(last_msg, list)
        assert any("CONTEXT" in block.get("text", "") for block in last_msg)

    # ── Round-trip conversion tests ─────────────────────────

    def test_roundtrip_simple_message(self):
        """Anthropic → OpenAI → Anthropic response should preserve semantics."""
        anthropic_body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "system": "Be helpful.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        openai_body = anthropic_messages_to_openai(anthropic_body)
        # Verify system was converted
        assert openai_body["messages"][0]["role"] == "system"
        assert openai_body["messages"][0]["content"] == "Be helpful."
        assert openai_body["messages"][1]["role"] == "user"

    def test_roundtrip_tool_exchange(self):
        """Full tool exchange should convert cleanly both ways."""
        anthropic_body = {
            "model": "test",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Fix auth.py"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me read the file."},
                        {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "auth.py"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "def login(): pass"}],
                },
            ],
        }
        openai_body = anthropic_messages_to_openai(anthropic_body)
        messages = openai_body["messages"]

        # User message preserved
        assert messages[0]["content"] == "Fix auth.py"

        # Assistant has text + tool_call
        assert messages[1]["role"] == "assistant"
        assert "Let me read the file." in (messages[1].get("content") or "")
        assert len(messages[1].get("tool_calls", [])) == 1

        # Tool result converted
        assert messages[2]["role"] == "tool"
        assert messages[2]["content"] == "def login(): pass"


# ── End-to-end: research-based enrichment injection ──────────


class TestResearchInjectionEndToEnd:
    """End-to-end validation that the mini research agent's structured output lands
    in the brain prompt as a sourced-findings block, not a raw snippets blob."""

    @pytest.mark.asyncio
    async def test_research_report_injected_as_structured_findings(self, monkeypatch):
        from smartsplit.models import (
            Mode,
            ResearchFinding,
            ResearchReport,
            TaskType,
            TokenUsage,
        )
        from smartsplit.triage import enrichment

        # Fake research output — what run_research would return on the happy path
        report = ResearchReport(
            findings=[
                ResearchFinding(
                    fact="FastAPI supports async natively",
                    source_url="https://fastapi.tiangolo.com/async",
                    confidence="high",
                ),
                ResearchFinding(
                    fact="Flask 3.0 adds native async views",
                    source_url="https://flask.palletsprojects.com/changelog",
                    confidence="medium",
                ),
            ],
            gaps=["no throughput benchmarks published"],
            queries_used=["fastapi flask async 2025", "flask 3 async support"],
        )

        monkeypatch.setattr(
            enrichment,
            "run_research",
            AsyncMock(return_value=report),
        )

        # Build a minimal ctx
        ctx = MagicMock()
        ctx.registry.brain_name = "groq"
        ctx.registry.call_brain = AsyncMock(return_value=("final answer", TokenUsage()))
        ctx.mode = Mode.BALANCED
        # router.route shouldn't be called for web_search-only enrichment, but must exist
        ctx.router.route = AsyncMock()

        original_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Should I pick FastAPI or Flask for an async API?"},
        ]

        content, results = await enrichment.enrich_and_forward(
            ctx,
            "Should I pick FastAPI or Flask for an async API?",
            ["web_search"],
            messages=original_messages,
        )

        assert content == "final answer"

        # The brain was called with enriched messages — inspect them
        call_kwargs = ctx.registry.call_brain.call_args.kwargs
        enriched = call_kwargs["messages"]
        last_user = enriched[-1]
        assert last_user["role"] == "user"

        injected = last_user["content"]
        # Unified structured injection format — section labelled "Research findings"
        assert "**Research findings**" in injected
        # Each fact cited with its URL + confidence
        assert "FACT (high): FastAPI supports async natively" in injected
        assert "https://fastapi.tiangolo.com/async" in injected
        assert "FACT (medium): Flask 3.0 adds native async views" in injected
        # Gaps surfaced
        assert "no throughput benchmarks published" in injected
        # Queries used are surfaced for observability
        assert "fastapi flask async 2025" in injected

        # worker RouteResult + brain result both returned
        assert len(results) == 2
        assert results[0].type == TaskType.WEB_SEARCH
        assert results[0].provider == "smartsplit.research"
        assert results[-1].provider == "groq"

    @pytest.mark.asyncio
    async def test_research_degraded_falls_back_to_raw_snippets_in_injection(self, monkeypatch):
        """When research degrades (returns a raw snippets string), the snippets still reach the brain."""
        from smartsplit.models import Mode, TokenUsage
        from smartsplit.triage import enrichment

        raw_snippets = "**Result 1**\nSome snippet text\nhttps://example.com/a"

        monkeypatch.setattr(
            enrichment,
            "run_research",
            AsyncMock(return_value=raw_snippets),
        )

        ctx = MagicMock()
        ctx.registry.brain_name = "groq"
        ctx.registry.call_brain = AsyncMock(return_value=("ok", TokenUsage()))
        ctx.mode = Mode.BALANCED
        ctx.router.route = AsyncMock()

        await enrichment.enrich_and_forward(
            ctx,
            "some prompt",
            ["web_search"],
            messages=[{"role": "user", "content": "some prompt"}],
        )

        enriched = ctx.registry.call_brain.call_args.kwargs["messages"]
        injected = enriched[-1]["content"]
        assert "Some snippet text" in injected
        assert "example.com" in injected
