"""Tests for the enrichment pipeline — _build_enrichment_subtasks, _build_enriched_messages, enrich_and_forward."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smartsplit.models import (
    Complexity,
    Mode,
    RouteResult,
    TaskType,
    TerminationState,
    TokenUsage,
)
from smartsplit.proxy.formats import anthropic_has_tool_named
from smartsplit.triage.enrichment import (
    _build_enriched_messages,
    _build_enrichment_subtasks,
    enrich_and_forward,
)

# ── _build_enrichment_subtasks ──────────────────────────────────


class TestBuildEnrichmentSubtasks:
    def test_web_search_creates_web_search_subtask(self):
        result = _build_enrichment_subtasks("find python docs", ["web_search"])
        assert len(result) == 1
        assert result[0].type == TaskType.WEB_SEARCH
        assert result[0].complexity == Complexity.LOW
        assert result[0].content == "find python docs"

    def test_pre_analysis_creates_reasoning_subtask(self):
        result = _build_enrichment_subtasks("explain decorators", ["pre_analysis"])
        assert len(result) == 1
        assert result[0].type == TaskType.REASONING
        assert result[0].complexity == Complexity.MEDIUM
        assert "explain decorators" in result[0].content

    def test_multi_perspective_creates_reasoning_subtask(self):
        result = _build_enrichment_subtasks("React vs Vue", ["multi_perspective"])
        assert len(result) == 1
        assert result[0].type == TaskType.REASONING
        assert result[0].complexity == Complexity.MEDIUM
        assert "React vs Vue" in result[0].content

    def test_context_summary_creates_summarize_subtask_and_truncates(self):
        long_content = "x" * 500
        messages = [
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": "short reply"},
        ]
        result = _build_enrichment_subtasks("summarize", ["context_summary"], messages=messages)
        assert len(result) == 1
        assert result[0].type == TaskType.SUMMARIZE
        assert result[0].complexity == Complexity.LOW
        # The long message should be truncated to 200 chars in the content
        assert long_content[:200] in result[0].content
        assert long_content[:201] not in result[0].content

    def test_unknown_enrichment_type_is_skipped(self):
        result = _build_enrichment_subtasks("hello", ["nonexistent_type"])
        assert result == []

    def test_empty_enrichment_types_returns_empty(self):
        result = _build_enrichment_subtasks("hello", [])
        assert result == []

    def test_multiple_enrichment_types_create_multiple_subtasks(self):
        result = _build_enrichment_subtasks("query", ["web_search", "pre_analysis", "multi_perspective"])
        assert len(result) == 3
        types = [s.type for s in result]
        assert types == [TaskType.WEB_SEARCH, TaskType.REASONING, TaskType.REASONING]


# ── _build_enriched_messages ────────────────────────────────────


class TestBuildEnrichedMessages:
    def _make_result(
        self,
        response: str = "some result",
        termination: TerminationState = TerminationState.COMPLETED,
        task_type: TaskType = TaskType.WEB_SEARCH,
    ) -> RouteResult:
        return RouteResult(
            type=task_type,
            response=response,
            provider="groq",
            termination=termination,
        )

    def test_injects_context_into_last_user_message(self):
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "my question"},
        ]
        results = [self._make_result("web info")]
        enriched = _build_enriched_messages(messages, "my question", results)
        assert len(enriched) == 2
        assert "[Additional context gathered by SmartSplit" in enriched[1]["content"]
        assert "Web Search: web info" in enriched[1]["content"]
        # Original prompt still present
        assert enriched[1]["content"].startswith("my question")

    def test_returns_original_when_no_results(self):
        messages = [{"role": "user", "content": "hello"}]
        result = _build_enriched_messages(messages, "hello", [])
        assert result is messages

    def test_skips_non_completed_results(self):
        messages = [{"role": "user", "content": "hello"}]
        results = [self._make_result("fail", termination=TerminationState.ALL_FAILED)]
        enriched = _build_enriched_messages(messages, "hello", results)
        # Non-completed results should be skipped, so messages returned as-is
        assert enriched is messages

    def test_skips_empty_response(self):
        messages = [{"role": "user", "content": "hello"}]
        results = [self._make_result("")]
        enriched = _build_enriched_messages(messages, "hello", results)
        # Empty response is skipped; RouteResult requires response to be str,
        # but the filter `if r.response` catches empty strings
        assert enriched is messages

    def test_deep_copies_original_messages(self):
        messages = [{"role": "user", "content": "original"}]
        results = [self._make_result("context")]
        enriched = _build_enriched_messages(messages, "original", results)
        # enriched should be modified
        assert "[Additional context" in enriched[0]["content"]
        # original must NOT be mutated
        assert messages[0]["content"] == "original"

    def test_appends_user_message_when_none_exists(self):
        messages = [{"role": "system", "content": "system only"}]
        results = [self._make_result("context")]
        enriched = _build_enriched_messages(messages, "prompt", results)
        assert len(enriched) == 2
        assert enriched[-1]["role"] == "user"
        assert "[Additional context" in enriched[-1]["content"]


# ── enrich_and_forward ──────────────────────────────────────────


def _make_ctx(
    brain_side_effect=None,
    brain_return=None,
    free_llm_return='["python docs"]',
    route_return=None,
):
    """Build a mock ProxyContext for enrich_and_forward tests."""
    ctx = MagicMock()
    ctx.registry.brain_name = "groq"
    ctx.mode = Mode.BALANCED

    # call_free_llm — used for search query extraction
    ctx.registry.call_free_llm = AsyncMock(return_value=free_llm_return)

    # call_brain
    ctx.registry.call_brain = AsyncMock()
    if brain_side_effect:
        ctx.registry.call_brain.side_effect = brain_side_effect
    else:
        ctx.registry.call_brain.return_value = brain_return or ("brain answer", TokenUsage())

    # router.route — returns a RouteResult per subtask
    if route_return is None:
        route_return = RouteResult(
            type=TaskType.WEB_SEARCH,
            response="search result",
            provider="serper",
            termination=TerminationState.COMPLETED,
        )
    ctx.router.route = AsyncMock(return_value=route_return)

    return ctx


class TestEnrichAndForward:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        ctx = _make_ctx(brain_return=("final answer", TokenUsage()))
        content, results = await enrich_and_forward(ctx, "search python docs", ["web_search"])
        assert content == "final answer"
        # results = worker results + brain result
        assert len(results) == 2
        assert results[-1].type == TaskType.GENERAL
        assert results[-1].provider == "groq"
        assert results[-1].termination == TerminationState.COMPLETED

    @pytest.mark.asyncio
    async def test_brain_failure_retries_without_enrichment(self):
        # First call_brain fails, second succeeds
        ctx = _make_ctx(
            brain_side_effect=[
                RuntimeError("enriched failed"),
                ("fallback answer", TokenUsage()),
            ]
        )
        content, results = await enrich_and_forward(ctx, "search python docs", ["web_search"])
        assert content == "fallback answer"
        assert ctx.registry.call_brain.call_count == 2
        # Brain result should be ESCALATED
        assert results[-1].termination == TerminationState.ESCALATED

    @pytest.mark.asyncio
    async def test_both_brain_attempts_fail_returns_none(self):
        ctx = _make_ctx(
            brain_side_effect=[
                RuntimeError("first fail"),
                RuntimeError("second fail"),
            ]
        )
        content, results = await enrich_and_forward(ctx, "search python docs", ["web_search"])
        assert content is None
        assert results == []


# ── anthropic_has_tool_named ──────────────────────────────────


class TestAnthropicHasToolNamed:
    def test_matches_present_tool(self):
        body = {
            "tools": [
                {"name": "web_search", "description": "Search the web", "input_schema": {}},
                {"name": "Read", "description": "Read a file", "input_schema": {}},
            ]
        }
        assert anthropic_has_tool_named(body, "web_search") is True
        assert anthropic_has_tool_named(body, "Read") is True
        assert anthropic_has_tool_named(body, "web_search", "Read") is True

    def test_does_not_match_absent_tool(self):
        body = {
            "tools": [
                {"name": "Read", "description": "Read a file", "input_schema": {}},
            ]
        }
        assert anthropic_has_tool_named(body, "web_search") is False
        assert anthropic_has_tool_named(body, "nonexistent") is False

    def test_no_tools_returns_false(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}
        assert anthropic_has_tool_named(body, "web_search") is False

    def test_empty_tools_list_returns_false(self):
        body = {"tools": []}
        assert anthropic_has_tool_named(body, "web_search") is False


# ── Web search cascade (process_anthropic_request_lite) ────────


def _make_pipeline_ctx(
    has_search_provider: bool = False,
    enrich_only_return: list[RouteResult] | None = None,
    detect_return: tuple[str, list[str]] | None = None,
    enrichment_skip_until: float = 0.0,
) -> MagicMock:
    """Build a mock ProxyContext for pipeline tests."""

    ctx = MagicMock()
    ctx.enabled = True
    ctx.enrichment_skip_until = enrichment_skip_until
    ctx.detector = None  # disable fake tool_use path
    ctx.anticipator = None
    ctx.pattern_learner = None
    ctx.mode = Mode.BALANCED

    # Search providers
    if has_search_provider:
        ctx.registry.get_search_providers.return_value = {"serper": MagicMock()}
    else:
        ctx.registry.get_search_providers.return_value = {}

    # Anticipation stats
    ctx.anticipation_stats = {
        "requests_with_tools": 0,
        "predictions_made": 0,
        "predictions_skipped": 0,
        "tools_anticipated": 0,
        "tools_injected": 0,
        "tools_failed": 0,
        "files_from_regex": 0,
        "files_already_read": 0,
        "files_recently_written": 0,
    }

    return ctx


class TestWebSearchCascade:
    @pytest.mark.asyncio
    async def test_web_search_skipped_no_provider_no_agent_tool(self):
        """web_search enrichment removed when no search provider AND no agent tool."""
        from smartsplit.proxy.pipeline import process_anthropic_request_lite
        from smartsplit.triage.detector import TriageDecision

        ctx = _make_pipeline_ctx(has_search_provider=False)

        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "what are the latest Python 3.13 features"}],
            # No web_search tool in agent's tools
            "tools": [{"name": "Read", "description": "Read a file", "input_schema": {}}],
        }

        # Force detect to return ENRICH with web_search
        with patch("smartsplit.proxy.pipeline.detect", return_value=(TriageDecision.ENRICH, ["web_search"])):
            result = await process_anthropic_request_lite(ctx, body, {})

        # web_search was the only enrichment type, and it was removed → TRANSPARENT
        assert result["type"] == "passthrough"

    @pytest.mark.asyncio
    async def test_cascade_serper_fails_returns_fake_tool_use(self):
        """When web_search enrichment fails but agent has the tool, return FAKE tool_use."""
        from smartsplit.proxy.pipeline import process_anthropic_request_lite
        from smartsplit.triage.detector import TriageDecision

        ctx = _make_pipeline_ctx(has_search_provider=True)
        # Set the extracted search query
        ctx.last_search_query = "Python 3.13 new features"

        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "what are the latest Python 3.13 features"}],
            "tools": [
                {"name": "web_search", "description": "Search the web", "input_schema": {}},
                {"name": "Read", "description": "Read a file", "input_schema": {}},
            ],
        }

        # enrich_only returns empty result for web_search (Serper failed)
        empty_results: list[RouteResult] = []

        with (
            patch("smartsplit.proxy.pipeline.detect", return_value=(TriageDecision.ENRICH, ["web_search"])),
            patch("smartsplit.proxy.pipeline.enrich_only", AsyncMock(return_value=empty_results)),
        ):
            result = await process_anthropic_request_lite(ctx, body, {})

        assert result["type"] == "fake"
        # Verify it's a tool_use response with the search query
        fake_body = result["body"]
        assert fake_body["stop_reason"] == "tool_use"
        tool_block = fake_body["content"][0]
        assert tool_block["type"] == "tool_use"
        assert tool_block["name"] == "web_search"
        assert tool_block["input"]["query"] == "Python 3.13 new features"

    @pytest.mark.asyncio
    async def test_cascade_uses_WebSearch_name_when_available(self):
        """Cascade picks 'WebSearch' tool name when that's what the agent has."""
        from smartsplit.proxy.pipeline import process_anthropic_request_lite
        from smartsplit.triage.detector import TriageDecision

        ctx = _make_pipeline_ctx(has_search_provider=True)
        ctx.last_search_query = "test query"

        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "search for something"}],
            "tools": [{"name": "WebSearch", "description": "Search", "input_schema": {}}],
        }

        with (
            patch("smartsplit.proxy.pipeline.detect", return_value=(TriageDecision.ENRICH, ["web_search"])),
            patch("smartsplit.proxy.pipeline.enrich_only", AsyncMock(return_value=[])),
        ):
            result = await process_anthropic_request_lite(ctx, body, {})

        assert result["type"] == "fake"
        assert result["body"]["content"][0]["name"] == "WebSearch"


# ── Enrichment backoff ────────────────────────────────────────


class TestEnrichmentBackoff:
    @pytest.mark.asyncio
    async def test_enrichment_skipped_during_backoff(self):
        """Enrichment is skipped when enrichment_skip_until is in the future."""
        from smartsplit.proxy.pipeline import process_anthropic_request_lite

        ctx = _make_pipeline_ctx(enrichment_skip_until=time.time() + 300)

        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "what are the latest Python 3.13 features"}],
        }

        # Even if detect would say ENRICH, it should be skipped due to backoff
        with patch("smartsplit.proxy.pipeline.detect") as mock_detect:
            result = await process_anthropic_request_lite(ctx, body, {})

        # detect should never be called — the enrichment block is skipped entirely
        mock_detect.assert_not_called()
        assert result["type"] == "passthrough"

    @pytest.mark.asyncio
    async def test_enrichment_runs_after_backoff_expires(self):
        """Enrichment runs normally when enrichment_skip_until is in the past."""
        from smartsplit.proxy.pipeline import process_anthropic_request_lite
        from smartsplit.triage.detector import TriageDecision

        ctx = _make_pipeline_ctx(
            enrichment_skip_until=time.time() - 10,  # expired 10s ago
            has_search_provider=True,
        )

        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "what are the latest Python 3.13 features"}],
        }

        web_result = RouteResult(
            type=TaskType.WEB_SEARCH,
            response="Python 3.13 info here",
            provider="serper",
            termination=TerminationState.COMPLETED,
        )

        with (
            patch("smartsplit.proxy.pipeline.detect", return_value=(TriageDecision.ENRICH, ["web_search"])),
            patch("smartsplit.proxy.pipeline.enrich_only", AsyncMock(return_value=[web_result])),
        ):
            result = await process_anthropic_request_lite(ctx, body, {})

        # Enrichment ran and injected context → body was modified
        assert result["type"] == "modified"
        # Verify the enrichment was injected into the last user message
        last_user = result["body"]["messages"][-1]
        assert "Python 3.13 info here" in str(last_user["content"])
