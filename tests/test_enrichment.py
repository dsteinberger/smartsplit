"""Tests for the enrichment pipeline — _build_enrichment_subtasks, build_enriched_messages, enrich_and_forward."""

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
    _build_enrichment_subtasks,
    build_enriched_messages,
    enrich_and_forward,
)

# ── _build_enrichment_subtasks ──────────────────────────────────


class TestBuildEnrichmentSubtasks:
    def test_web_search_is_not_handled_here(self):
        """web_search is intercepted upstream and routed to the mini research agent —
        _build_enrichment_subtasks must ignore it."""
        result = _build_enrichment_subtasks("find python docs", ["web_search"])
        assert result == []

    def test_pre_analysis_code_prompt_uses_code_template_at_high_complexity(self):
        # "refactor" + "function" → code domain → specialized code template
        result = _build_enrichment_subtasks("Please refactor this function and explain", ["pre_analysis"])
        assert len(result) == 1
        assert result[0].type == TaskType.REASONING
        assert result[0].complexity == Complexity.HIGH
        assert "Invariants to preserve" in result[0].content
        assert "refactor" in result[0].content

    def test_pre_analysis_neutral_prompt_falls_back_to_medium_complexity(self):
        # No strong domain keywords → generic fallback + MEDIUM
        result = _build_enrichment_subtasks("Please help me", ["pre_analysis"])
        assert len(result) == 1
        assert result[0].type == TaskType.REASONING
        assert result[0].complexity == Complexity.MEDIUM
        assert "Key concepts" in result[0].content

    def test_multi_perspective_code_prompt_uses_code_template_at_high_complexity(self):
        # React vs Vue → code domain → specialized code template
        result = _build_enrichment_subtasks("React vs Vue for a large-scale javascript frontend", ["multi_perspective"])
        assert len(result) == 1
        assert result[0].type == TaskType.REASONING
        assert result[0].complexity == Complexity.HIGH
        assert "**Claim**" in result[0].content
        assert "**Evidence**" in result[0].content

    def test_multi_perspective_neutral_prompt_falls_back_to_medium(self):
        result = _build_enrichment_subtasks("Choose one", ["multi_perspective"])
        assert len(result) == 1
        assert result[0].complexity == Complexity.MEDIUM
        assert "claim, evidence, cost or risk" in result[0].content

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
        # web_search is filtered out (handled by research pipeline instead)
        result = _build_enrichment_subtasks("query", ["web_search", "pre_analysis", "multi_perspective"])
        assert len(result) == 2
        types = [s.type for s in result]
        assert types == [TaskType.REASONING, TaskType.REASONING]


# ── build_enriched_messages ────────────────────────────────────


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
        enriched = build_enriched_messages(messages, "my question", results)
        assert len(enriched) == 2
        injected = enriched[1]["content"]
        assert "[Additional context gathered by SmartSplit" in injected
        # New unified format: bold section label + separators, worker content preserved
        assert "**Research findings**" in injected
        assert "web info" in injected
        assert "---" in injected
        # Original prompt still present
        assert injected.startswith("my question")

    def test_reasoning_result_uses_analysis_label(self):
        messages = [{"role": "user", "content": "hi"}]
        results = [self._make_result("## Invariants\n- X", task_type=TaskType.REASONING)]
        enriched = build_enriched_messages(messages, "hi", results)
        assert "**Analysis**" in enriched[0]["content"]
        # Worker markdown headings are preserved, not flattened to a bullet
        assert "## Invariants" in enriched[0]["content"]

    def test_multiple_results_each_get_own_section(self):
        messages = [{"role": "user", "content": "hi"}]
        results = [
            self._make_result("findings content", task_type=TaskType.WEB_SEARCH),
            self._make_result("## Positions\n- A", task_type=TaskType.REASONING),
        ]
        enriched = build_enriched_messages(messages, "hi", results)
        injected = enriched[0]["content"]
        # Two labelled sections
        assert injected.count("---") >= 3  # opening + between + closing
        assert "**Research findings**" in injected
        assert "**Analysis**" in injected

    def test_returns_original_when_no_results(self):
        messages = [{"role": "user", "content": "hello"}]
        result = build_enriched_messages(messages, "hello", [])
        assert result is messages

    def test_skips_non_completed_results(self):
        messages = [{"role": "user", "content": "hello"}]
        results = [self._make_result("fail", termination=TerminationState.ALL_FAILED)]
        enriched = build_enriched_messages(messages, "hello", results)
        # Non-completed results should be skipped, so messages returned as-is
        assert enriched is messages

    def test_skips_empty_response(self):
        messages = [{"role": "user", "content": "hello"}]
        results = [self._make_result("")]
        enriched = build_enriched_messages(messages, "hello", results)
        # Empty response is skipped; RouteResult requires response to be str,
        # but the filter `if r.response` catches empty strings
        assert enriched is messages

    def test_deep_copies_original_messages(self):
        messages = [{"role": "user", "content": "original"}]
        results = [self._make_result("context")]
        enriched = build_enriched_messages(messages, "original", results)
        # enriched should be modified
        assert "[Additional context" in enriched[0]["content"]
        # original must NOT be mutated
        assert messages[0]["content"] == "original"

    def test_appends_user_message_when_none_exists(self):
        messages = [{"role": "system", "content": "system only"}]
        results = [self._make_result("context")]
        enriched = build_enriched_messages(messages, "prompt", results)
        assert len(enriched) == 2
        assert enriched[-1]["role"] == "user"
        assert "[Additional context" in enriched[-1]["content"]


# ── enrich_and_forward ──────────────────────────────────────────


def _make_ctx(
    brain_side_effect=None,
    brain_return=None,
    worker_llm_return='["python docs"]',
    route_return=None,
):
    """Build a mock ProxyContext for enrich_and_forward tests."""
    ctx = MagicMock()
    ctx.registry.brain_name = "groq"
    ctx.mode = Mode.BALANCED

    # call_worker_llm — used for search query extraction
    ctx.registry.call_worker_llm = AsyncMock(return_value=worker_llm_return)

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
        # Use pre_analysis (non-research path) to exercise the router.route branch
        ctx = _make_ctx(
            brain_return=("final answer", TokenUsage()),
            route_return=RouteResult(
                type=TaskType.REASONING,
                response="pre-analysis result",
                provider="groq",
                termination=TerminationState.COMPLETED,
            ),
        )
        content, results = await enrich_and_forward(ctx, "explain decorators please", ["pre_analysis"])
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
            ],
            route_return=RouteResult(
                type=TaskType.REASONING,
                response="analysis",
                provider="groq",
                termination=TerminationState.COMPLETED,
            ),
        )
        content, results = await enrich_and_forward(ctx, "analyze this code please", ["pre_analysis"])
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
            ],
            route_return=RouteResult(
                type=TaskType.REASONING,
                response="analysis",
                provider="groq",
                termination=TerminationState.COMPLETED,
            ),
        )
        content, results = await enrich_and_forward(ctx, "analyze this please", ["pre_analysis"])
        assert content is None
        assert results == []


# ── web_search → mini research agent integration ──────────────


class TestEnrichAndForwardWebSearch:
    """Tests for the web_search → run_research integration in enrich_and_forward/enrich_only."""

    @pytest.mark.asyncio
    async def test_web_search_routes_to_research_pipeline(self):
        """web_search enrichment calls run_research instead of router.route."""
        from smartsplit.models import ResearchFinding, ResearchReport
        from smartsplit.triage.enrichment import enrich_only

        report = ResearchReport(
            findings=[ResearchFinding(fact="F1", source_url="https://a", confidence="high")],
            gaps=[],
            queries_used=["q1"],
        )

        ctx = _make_ctx()
        with patch("smartsplit.triage.enrichment.run_research", AsyncMock(return_value=report)):
            results = await enrich_only(ctx, "find LLM routers", ["web_search"])

        assert len(results) == 1
        assert results[0].type == TaskType.WEB_SEARCH
        assert results[0].provider == "smartsplit.research"
        # Formatted report contains the sourced fact
        assert "FACT (high): F1" in results[0].response
        assert "https://a" in results[0].response

    @pytest.mark.asyncio
    async def test_web_search_degraded_returns_raw_snippets(self):
        """When research degrades to a raw-snippet string, it's still wrapped as a RouteResult."""
        from smartsplit.triage.enrichment import enrich_only

        ctx = _make_ctx()
        with patch(
            "smartsplit.triage.enrichment.run_research",
            AsyncMock(return_value="raw snippet content"),
        ):
            results = await enrich_only(ctx, "find something", ["web_search"])

        assert len(results) == 1
        assert results[0].response == "raw snippet content"

    @pytest.mark.asyncio
    async def test_web_search_empty_returns_no_result(self):
        """When research produces nothing usable (empty string), no RouteResult is emitted."""
        from smartsplit.triage.enrichment import enrich_only

        ctx = _make_ctx()
        with patch("smartsplit.triage.enrichment.run_research", AsyncMock(return_value="")):
            results = await enrich_only(ctx, "find something", ["web_search"])

        assert results == []

    @pytest.mark.asyncio
    async def test_web_search_and_pre_analysis_cohabit(self):
        """web_search goes through research, pre_analysis through router.route — both emit results."""
        from smartsplit.models import ResearchFinding, ResearchReport
        from smartsplit.triage.enrichment import enrich_only

        report = ResearchReport(
            findings=[ResearchFinding(fact="F", source_url="https://u", confidence="medium")],
            gaps=[],
            queries_used=["q"],
        )

        ctx = _make_ctx(
            route_return=RouteResult(
                type=TaskType.REASONING,
                response="analysis output",
                provider="groq",
                termination=TerminationState.COMPLETED,
            ),
        )
        with patch("smartsplit.triage.enrichment.run_research", AsyncMock(return_value=report)):
            results = await enrich_only(
                ctx,
                "compare foo and bar in depth please",
                ["web_search", "pre_analysis"],
            )

        types = [r.type for r in results]
        assert TaskType.WEB_SEARCH in types
        assert TaskType.REASONING in types
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_research_disabled_via_config_skips_web_search(self):
        """When cfg.research_enabled is False, web_search enrichment is skipped entirely."""
        from smartsplit.config import SmartSplitConfig
        from smartsplit.triage.enrichment import enrich_only

        ctx = _make_ctx()
        ctx.config = SmartSplitConfig(research_enabled=False)
        research_mock = AsyncMock(return_value="would be used")
        with patch("smartsplit.triage.enrichment.run_research", research_mock):
            results = await enrich_only(ctx, "find things", ["web_search"])

        assert results == []
        research_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_research_budget_from_config_passed_to_run_research(self):
        """cfg.research_budget_seconds is forwarded as run_research's total_budget."""
        from smartsplit.config import SmartSplitConfig
        from smartsplit.triage.enrichment import enrich_only

        ctx = _make_ctx()
        ctx.config = SmartSplitConfig(research_budget_seconds=3.5)

        captured = {}

        async def fake(ctx_, prompt, messages, **kwargs):
            captured.update(kwargs)
            return ""

        with patch("smartsplit.triage.enrichment.run_research", side_effect=fake):
            await enrich_only(ctx, "find things", ["web_search"])

        assert captured.get("total_budget") == 3.5

    @pytest.mark.asyncio
    async def test_research_crash_doesnt_break_enrichment(self):
        """If run_research raises, we log and continue with other enrichments."""
        from smartsplit.triage.enrichment import enrich_only

        ctx = _make_ctx(
            route_return=RouteResult(
                type=TaskType.REASONING,
                response="analysis",
                provider="groq",
                termination=TerminationState.COMPLETED,
            ),
        )
        with patch(
            "smartsplit.triage.enrichment.run_research",
            AsyncMock(side_effect=RuntimeError("pipeline exploded")),
        ):
            results = await enrich_only(
                ctx,
                "compare foo and bar in detail",
                ["web_search", "pre_analysis"],
            )

        # Research crashed → no web_search result, but pre_analysis still ran
        assert len(results) == 1
        assert results[0].type == TaskType.REASONING


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


class TestResearchQueryStoredOnCtx:
    """Regression: enrich_only must expose the planned search query on ctx.last_search_query
    so the FAKE tool_use fallback (proxy mode) can still surface a reasonable query when
    Serper is down."""

    @pytest.mark.asyncio
    async def test_store_query_on_ctx_forwarded_to_run_research(self):
        from smartsplit.triage.enrichment import enrich_only

        ctx = _make_ctx()

        captured_kwargs = {}

        async def fake_research(ctx_, prompt, messages, **kwargs):
            captured_kwargs.update(kwargs)
            return ""

        with patch("smartsplit.triage.enrichment.run_research", side_effect=fake_research):
            await enrich_only(ctx, "find things", ["web_search"])

        assert captured_kwargs.get("store_query_on_ctx") is True

    @pytest.mark.asyncio
    async def test_enrich_and_forward_does_not_store_query_on_ctx(self):
        """API mode (enrich_and_forward) does not need the FAKE fallback query stored."""
        from smartsplit.triage.enrichment import enrich_and_forward

        ctx = _make_ctx(brain_return=("ok", TokenUsage()))

        captured_kwargs = {}

        async def fake_research(ctx_, prompt, messages, **kwargs):
            captured_kwargs.update(kwargs)
            return ""

        with patch("smartsplit.triage.enrichment.run_research", side_effect=fake_research):
            await enrich_and_forward(ctx, "find things", ["web_search"])

        assert captured_kwargs.get("store_query_on_ctx") is False
