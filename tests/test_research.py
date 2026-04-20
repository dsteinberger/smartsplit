"""Tests for the mini research agent (smartsplit/triage/research.py)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from smartsplit.models import ResearchFinding, ResearchReport  # noqa: F401
from smartsplit.triage.research import (
    Budget,
    _fallback_queries,
    gap_fill,
    plan_queries,
    read_and_synthesize,
    run_research,
    search_parallel,
)

# ── Budget ──────────────────────────────────────────────────


class TestBudget:
    def test_fresh_budget_has_full_time(self):
        b = Budget(5.0)
        assert b.has_at_least(4.9)
        assert b.remaining() <= 5.0

    def test_child_returns_fraction(self):
        b = Budget(10.0)
        child = b.child(0.3)
        assert 2.5 <= child <= 3.0  # allow tiny timing slop

    @pytest.mark.asyncio
    async def test_expired_budget_returns_zero(self):
        b = Budget(0.05)
        await asyncio.sleep(0.1)
        assert b.remaining() == 0.0
        assert not b.has_at_least(0.01)
        assert b.child(1.0) == 0.0


# ── plan_queries ────────────────────────────────────────────


def _ctx_with_worker(response: str) -> MagicMock:
    ctx = MagicMock()
    ctx.registry.call_worker_llm = AsyncMock(return_value=response)
    return ctx


class TestPlanQueries:
    @pytest.mark.asyncio
    async def test_parses_json_array(self):
        ctx = _ctx_with_worker('["llm routing 2025", "proxy github examples"]')
        result = await plan_queries(ctx, "find LLM routers", [], budget=Budget(5.0))
        assert result == ["llm routing 2025", "proxy github examples"]

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        ctx = _ctx_with_worker('```json\n["q1", "q2"]\n```')
        result = await plan_queries(ctx, "prompt", [], budget=Budget(5.0))
        assert result == ["q1", "q2"]

    @pytest.mark.asyncio
    async def test_caps_at_three_queries(self):
        ctx = _ctx_with_worker('["a","b","c","d","e"]')
        result = await plan_queries(ctx, "p", [], budget=Budget(5.0))
        assert len(result) == 3
        assert result == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_falls_back_on_invalid_json(self):
        ctx = _ctx_with_worker("this is not JSON at all")
        result = await plan_queries(ctx, "my raw prompt", [], budget=Budget(5.0))
        assert result == ["my raw prompt"]

    @pytest.mark.asyncio
    async def test_falls_back_on_empty_array(self):
        ctx = _ctx_with_worker("[]")
        result = await plan_queries(ctx, "my raw prompt", [], budget=Budget(5.0))
        assert result == ["my raw prompt"]

    @pytest.mark.asyncio
    async def test_falls_back_on_non_string_items(self):
        ctx = _ctx_with_worker("[1, 2, 3]")
        result = await plan_queries(ctx, "my raw prompt", [], budget=Budget(5.0))
        assert result == ["my raw prompt"]

    @pytest.mark.asyncio
    async def test_falls_back_on_worker_failure(self):
        ctx = MagicMock()
        ctx.registry.call_worker_llm = AsyncMock(side_effect=RuntimeError("no workers"))
        result = await plan_queries(ctx, "my raw prompt", [], budget=Budget(5.0))
        assert result == ["my raw prompt"]

    @pytest.mark.asyncio
    async def test_falls_back_on_timeout(self):
        async def slow(*_args, **_kwargs):
            await asyncio.sleep(5.0)
            return "[]"

        ctx = MagicMock()
        ctx.registry.call_worker_llm = AsyncMock(side_effect=slow)
        # Tight budget forces timeout in wait_for
        result = await plan_queries(ctx, "my raw prompt", [], budget=Budget(1.2))
        assert result == ["my raw prompt"]

    @pytest.mark.asyncio
    async def test_skipped_when_budget_exhausted(self):
        ctx = MagicMock()
        ctx.registry.call_worker_llm = AsyncMock(return_value='["should not be called"]')
        b = Budget(0.05)
        await asyncio.sleep(0.1)  # exhaust
        result = await plan_queries(ctx, "raw", [], budget=b)
        assert result == ["raw"]
        ctx.registry.call_worker_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_prompt_returns_empty_list(self):
        ctx = _ctx_with_worker('["q"]')
        b = Budget(0.01)
        await asyncio.sleep(0.05)
        result = await plan_queries(ctx, "   ", [], budget=b)
        assert result == []


# ── search_parallel ────────────────────────────────────────


def _ctx_with_search(search_fn) -> MagicMock:
    """Build a ctx whose only search provider has ``search_fn`` as its search method."""
    ctx = MagicMock()
    provider = MagicMock()
    provider.search = search_fn
    ctx.registry.get_search_providers.return_value = {"serper": provider}
    return ctx


class TestSearchParallel:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_provider(self):
        ctx = MagicMock()
        ctx.registry.get_search_providers.return_value = {}
        result = await search_parallel(ctx, ["q1"], budget=Budget(5.0))
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_queries(self):
        ctx = _ctx_with_search(AsyncMock(return_value="results"))
        result = await search_parallel(ctx, [], budget=Budget(5.0))
        assert result == []

    @pytest.mark.asyncio
    async def test_runs_all_queries_and_returns_tuples(self):
        ctx = _ctx_with_search(AsyncMock(side_effect=["res1", "res2", "res3"]))
        result = await search_parallel(ctx, ["q1", "q2", "q3"], budget=Budget(5.0))
        assert len(result) == 3
        assert result[0] == ("q1", "res1")
        assert result[1] == ("q2", "res2")

    @pytest.mark.asyncio
    async def test_drops_failed_queries_keeps_others(self):
        async def mixed(query):
            if query == "bad":
                raise RuntimeError("boom")
            return f"ok_{query}"

        ctx = _ctx_with_search(mixed)
        result = await search_parallel(ctx, ["q1", "bad", "q2"], budget=Budget(5.0))
        assert len(result) == 2
        assert ("q1", "ok_q1") in result
        assert ("q2", "ok_q2") in result

    @pytest.mark.asyncio
    async def test_drops_empty_snippets(self):
        ctx = _ctx_with_search(AsyncMock(side_effect=["", "   ", "real"]))
        result = await search_parallel(ctx, ["q1", "q2", "q3"], budget=Budget(5.0))
        assert result == [("q3", "real")]

    @pytest.mark.asyncio
    async def test_skipped_when_budget_tight(self):
        search_mock = AsyncMock(return_value="result")
        ctx = _ctx_with_search(search_mock)
        b = Budget(0.05)
        await asyncio.sleep(0.1)
        result = await search_parallel(ctx, ["q1"], budget=b)
        assert result == []
        search_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_slow_query_times_out_others_continue(self):
        async def maybe_slow(query):
            if query == "slow":
                await asyncio.sleep(5.0)
                return "slow_result"
            return f"fast_{query}"

        ctx = _ctx_with_search(maybe_slow)
        # Budget is short so per_query timeout is short too
        result = await search_parallel(ctx, ["q1", "slow", "q2"], budget=Budget(2.5))
        queries_done = [q for q, _ in result]
        assert "q1" in queries_done
        assert "q2" in queries_done
        assert "slow" not in queries_done


# ── read_and_synthesize ────────────────────────────────────


_VALID_SYNTH = """{
  "findings": [
    {"fact": "Framework X supports streaming", "source_url": "https://github.com/x", "confidence": "high"},
    {"fact": "Framework Y is deprecated", "source_url": "https://y.io/news", "confidence": "medium"}
  ],
  "gaps": ["no performance benchmarks"]
}"""


class TestReadAndSynthesize:
    @pytest.mark.asyncio
    async def test_returns_empty_report_when_no_search_results(self):
        ctx = _ctx_with_worker("{}")
        report = await read_and_synthesize(ctx, "q", [], budget=Budget(5.0))
        assert report.findings == []
        assert report.queries_used == []
        ctx.registry.call_worker_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_parses_valid_json_into_report(self):
        ctx = _ctx_with_worker(_VALID_SYNTH)
        report = await read_and_synthesize(ctx, "question", [("q1", "snippets...")], budget=Budget(5.0))
        assert len(report.findings) == 2
        assert report.findings[0].fact == "Framework X supports streaming"
        assert report.findings[0].source_url == "https://github.com/x"
        assert report.findings[0].confidence == "high"
        assert report.gaps == ["no performance benchmarks"]
        assert report.queries_used == ["q1"]

    @pytest.mark.asyncio
    async def test_caps_findings_at_six(self):
        too_many = {
            "findings": [{"fact": f"f{i}", "source_url": f"https://u/{i}", "confidence": "medium"} for i in range(15)],
            "gaps": [],
        }
        ctx = _ctx_with_worker(json.dumps(too_many))
        report = await read_and_synthesize(ctx, "q", [("q1", "results")], budget=Budget(5.0))
        assert len(report.findings) == 6

    @pytest.mark.asyncio
    async def test_drops_findings_without_source_url(self):
        partial = {
            "findings": [
                {"fact": "has source", "source_url": "https://u", "confidence": "high"},
                {"fact": "no source", "confidence": "high"},
                {"fact": "empty source", "source_url": "", "confidence": "high"},
            ],
            "gaps": [],
        }
        ctx = _ctx_with_worker(json.dumps(partial))
        report = await read_and_synthesize(ctx, "q", [("q1", "r")], budget=Budget(5.0))
        assert len(report.findings) == 1
        assert report.findings[0].fact == "has source"

    @pytest.mark.asyncio
    async def test_falls_back_on_invalid_json(self):
        ctx = _ctx_with_worker("not json at all")
        report = await read_and_synthesize(ctx, "q", [("q1", "results")], budget=Budget(5.0))
        assert report.findings == []
        assert report.queries_used == ["q1"]
        assert "synthesis failed" in report.gaps[0]

    @pytest.mark.asyncio
    async def test_falls_back_when_zero_findings(self):
        empty = {"findings": [], "gaps": ["nothing useful"]}
        ctx = _ctx_with_worker(json.dumps(empty))
        report = await read_and_synthesize(ctx, "q", [("q1", "results")], budget=Budget(5.0))
        assert report.findings == []
        assert "synthesis failed" in report.gaps[0]

    @pytest.mark.asyncio
    async def test_falls_back_on_worker_exception(self):
        ctx = MagicMock()
        ctx.registry.call_worker_llm = AsyncMock(side_effect=RuntimeError("no workers"))
        report = await read_and_synthesize(ctx, "q", [("q1", "snippets")], budget=Budget(5.0))
        assert report.findings == []
        assert report.queries_used == ["q1"]

    @pytest.mark.asyncio
    async def test_skipped_when_budget_tight(self):
        worker = AsyncMock(return_value=_VALID_SYNTH)
        ctx = MagicMock()
        ctx.registry.call_worker_llm = worker
        b = Budget(0.05)
        await asyncio.sleep(0.1)
        report = await read_and_synthesize(ctx, "q", [("q1", "results")], budget=b)
        assert report.findings == []
        worker.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_confidence_defaults_to_medium(self):
        weird = {
            "findings": [
                {"fact": "x", "source_url": "https://u", "confidence": "absolutely"},
            ],
            "gaps": [],
        }
        ctx = _ctx_with_worker(json.dumps(weird))
        report = await read_and_synthesize(ctx, "q", [("q1", "r")], budget=Budget(5.0))
        assert report.findings[0].confidence == "medium"


# ── gap_fill ────────────────────────────────────────────────


def _report(findings=None, gaps=None, queries=None) -> ResearchReport:
    return ResearchReport(
        findings=findings or [],
        gaps=gaps or [],
        queries_used=queries or [],
    )


def _finding(fact: str, url: str = "https://u") -> ResearchFinding:
    return ResearchFinding(fact=fact, source_url=url, confidence="medium")


def _ctx_with_search_and_worker(search_fn, worker_responses: list[str]) -> MagicMock:
    """Build a ctx that has both a search provider and a worker LLM with queued responses."""
    ctx = MagicMock()
    provider = MagicMock()
    provider.search = search_fn
    ctx.registry.get_search_providers.return_value = {"serper": provider}
    ctx.registry.call_worker_llm = AsyncMock(side_effect=worker_responses)
    return ctx


class TestGapFill:
    @pytest.mark.asyncio
    async def test_skipped_when_no_gaps(self):
        ctx = MagicMock()
        ctx.registry.call_worker_llm = AsyncMock()
        report = _report(findings=[_finding("x")])
        result = await gap_fill(ctx, "prompt", report, budget=Budget(5.0))
        assert result is report or result == report
        ctx.registry.call_worker_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_skipped_when_synthesis_failed(self):
        ctx = MagicMock()
        ctx.registry.call_worker_llm = AsyncMock()
        report = _report(gaps=["synthesis failed — using raw snippets"])
        result = await gap_fill(ctx, "prompt", report, budget=Budget(5.0))
        assert result == report
        ctx.registry.call_worker_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_skipped_when_budget_tight(self):
        ctx = MagicMock()
        ctx.registry.call_worker_llm = AsyncMock()
        report = _report(gaps=["missing data"])
        b = Budget(0.05)
        await asyncio.sleep(0.1)
        result = await gap_fill(ctx, "prompt", report, budget=b)
        assert result == report
        ctx.registry.call_worker_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_original_when_gap_query_fails(self):
        ctx = MagicMock()
        provider = MagicMock()
        provider.search = AsyncMock(return_value="snippets")
        ctx.registry.get_search_providers.return_value = {"serper": provider}
        ctx.registry.call_worker_llm = AsyncMock(side_effect=RuntimeError("worker down"))
        report = _report(gaps=["missing data"])
        result = await gap_fill(ctx, "prompt", report, budget=Budget(5.0))
        # Worker down → no gap query → return original
        assert result == report

    @pytest.mark.asyncio
    async def test_merges_new_findings_into_report(self):
        # 1st worker call: GAP query planning → returns "extra query"
        # 2nd worker call: READ on the new snippet → returns a valid synthesis
        synth_response = json.dumps(
            {
                "findings": [{"fact": "extra fact", "source_url": "https://new.example", "confidence": "high"}],
                "gaps": [],
            }
        )
        ctx = _ctx_with_search_and_worker(
            AsyncMock(return_value="new_snippets"),
            worker_responses=['"extra query"', synth_response],
        )
        report = _report(
            findings=[_finding("original fact", "https://orig")],
            gaps=["missing benchmarks"],
            queries=["original query"],
        )
        result = await gap_fill(ctx, "prompt", report, budget=Budget(10.0))
        assert len(result.findings) == 2
        facts = [f.fact for f in result.findings]
        assert "original fact" in facts
        assert "extra fact" in facts
        assert "extra query" in result.queries_used

    @pytest.mark.asyncio
    async def test_accepts_bare_string_gap_query(self):
        synth_response = json.dumps(
            {
                "findings": [{"fact": "f", "source_url": "https://u", "confidence": "medium"}],
                "gaps": [],
            }
        )
        ctx = _ctx_with_search_and_worker(
            AsyncMock(return_value="snippets"),
            worker_responses=['"a bare query"', synth_response],
        )
        report = _report(gaps=["gap"])
        result = await gap_fill(ctx, "p", report, budget=Budget(10.0))
        assert "a bare query" in result.queries_used


# ── run_research orchestrator ──────────────────────────────


class TestRunResearch:
    @pytest.mark.asyncio
    async def test_full_happy_path(self):
        synth = json.dumps(
            {
                "findings": [
                    {"fact": "F1", "source_url": "https://a", "confidence": "high"},
                    {"fact": "F2", "source_url": "https://b", "confidence": "medium"},
                ],
                "gaps": [],
            }
        )
        ctx = _ctx_with_search_and_worker(
            AsyncMock(return_value="web snippets here"),
            worker_responses=['["q1", "q2"]', synth],
        )
        result = await run_research(ctx, "question", [], total_budget=10.0)
        assert isinstance(result, ResearchReport)
        assert len(result.findings) == 2
        assert result.queries_used == ["q1", "q2"]

    @pytest.mark.asyncio
    async def test_empty_when_no_search_provider(self):
        ctx = MagicMock()
        ctx.registry.get_search_providers.return_value = {}
        ctx.registry.call_worker_llm = AsyncMock(return_value='["q"]')
        result = await run_research(ctx, "question", [], total_budget=5.0)
        assert result == ""

    @pytest.mark.asyncio
    async def test_empty_when_search_fails(self):
        async def fail(_q):
            raise RuntimeError("boom")

        ctx = _ctx_with_search_and_worker(fail, worker_responses=['["q"]'])
        result = await run_research(ctx, "question", [], total_budget=5.0)
        assert result == ""

    @pytest.mark.asyncio
    async def test_degrades_to_raw_snippets_when_read_fails(self):
        # Plan succeeds, search succeeds, synth returns invalid JSON
        ctx = _ctx_with_search_and_worker(
            AsyncMock(return_value="raw snippet content"),
            worker_responses=['["q1"]', "not valid json"],
        )
        result = await run_research(ctx, "question", [], total_budget=10.0)
        assert isinstance(result, str)
        assert "raw snippet content" in result

    @pytest.mark.asyncio
    async def test_handles_plan_fallback(self):
        # Plan fails → fallback to raw prompt as single query
        synth = json.dumps(
            {
                "findings": [{"fact": "F", "source_url": "https://u", "confidence": "medium"}],
                "gaps": [],
            }
        )
        ctx = _ctx_with_search_and_worker(
            AsyncMock(return_value="snippets"),
            worker_responses=[RuntimeError("plan fail"), synth],
        )
        result = await run_research(ctx, "my specific question", [], total_budget=10.0)
        assert isinstance(result, ResearchReport)
        # Fallback query truncated from raw prompt
        assert "my specific question" in result.queries_used[0]


class TestFallbackQueries:
    def test_truncates_to_200_chars(self):
        long = "x" * 500
        assert _fallback_queries(long) == ["x" * 200]

    def test_empty_prompt_returns_empty_list(self):
        assert _fallback_queries("") == []
        assert _fallback_queries("   ") == []
