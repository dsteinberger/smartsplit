"""Integration tests — verify modules work together end-to-end."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from smartsplit.models import Complexity, Mode, RouteResult, Subtask, TaskType
from smartsplit.planner import Planner
from smartsplit.providers.registry import ProviderRegistry
from smartsplit.quota import QuotaTracker
from smartsplit.router import Router


class TestFullPipeline:
    """Test the complete decompose → route → synthesize pipeline with mocked HTTP."""

    @pytest.fixture
    def pipeline(self, make_config, tmp_path):
        """Build a complete pipeline with mocked providers."""
        config = make_config(["groq", "gemini", "serper", "anthropic"])
        http = httpx.AsyncClient()
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        registry = ProviderRegistry(config.providers, http)
        planner = Planner(registry)
        router = Router(registry, quota, config)

        # Mock all provider calls
        registry.get("groq").complete = AsyncMock(
            return_value='[{"type": "general", "content": "hello", "complexity": "low"}]'
        )
        registry.get("gemini").complete = AsyncMock(return_value="Gemini response")
        registry.get("anthropic").complete = AsyncMock(return_value="Claude response")
        registry.get("serper").search = AsyncMock(return_value="Search results")

        return {
            "config": config,
            "quota": quota,
            "registry": registry,
            "planner": planner,
            "router": router,
        }

    @pytest.mark.asyncio
    async def test_simple_query_decomposes_and_routes(self, pipeline):
        planner = pipeline["planner"]
        router = pipeline["router"]

        subtasks = await planner.decompose("What is Python?")
        assert len(subtasks) >= 1

        results = []
        for st in subtasks:
            result = await router.route(st)
            results.append(result)

        assert all(r.provider != "none" for r in results)
        assert all(r.response for r in results)

    @pytest.mark.asyncio
    async def test_search_subtask_routes_to_serper(self, pipeline):
        router = pipeline["router"]
        subtask = Subtask(type=TaskType.WEB_SEARCH, content="python tutorial", complexity=Complexity.LOW)
        result = await router.route(subtask)
        assert result.provider == "serper"
        assert result.response == "Search results"

    @pytest.mark.asyncio
    async def test_complex_code_routes_to_anthropic_in_quality_mode(self, pipeline):
        router = pipeline["router"]
        registry = pipeline["registry"]
        # Anthropic response must pass quality gate (>50 chars for high complexity)
        registry.get("anthropic").complete = AsyncMock(
            return_value="Here is a complex algorithm implementation with detailed explanation and working code."
        )
        subtask = Subtask(type=TaskType.CODE, content="complex algo", complexity=Complexity.HIGH)
        result = await router.route(subtask, mode=Mode.QUALITY)
        assert result.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_quota_updates_after_routing(self, pipeline):
        quota = pipeline["quota"]
        router = pipeline["router"]
        assert quota.get_usage("groq") == 0

        subtask = Subtask(type=TaskType.SUMMARIZE, content="summarize this", complexity=Complexity.LOW)
        await router.route(subtask)

        # Some provider was used — check total usage increased
        total = sum(quota.get_usage(p) for p in ("groq", "gemini", "anthropic", "serper"))
        assert total == 1

    @pytest.mark.asyncio
    async def test_synthesis_after_multiple_subtasks(self, pipeline):
        planner = pipeline["planner"]
        registry = pipeline["registry"]

        # Make planner return multiple subtasks
        # Multi-domain prompt (code + writing + web_search) triggers decomposition
        multi_domain_prompt = (
            "Search the latest features of React 19, then write a Python function "
            "implementing the best ones and draft a blog post about them"
        )
        registry.get("groq").complete = AsyncMock(
            side_effect=[
                # First call: decompose
                json.dumps(
                    [
                        {"type": "web_search", "content": "search this", "complexity": "low"},
                        {"type": "code", "content": "write code", "complexity": "high"},
                    ]
                ),
                # Second call: context injection
                "User wants React 19 features researched and implemented.",
                # Third call: synthesize
                "Here is the combined answer.",
            ]
        )

        subtasks = await planner.decompose(multi_domain_prompt)
        assert len(subtasks) == 2

        results = [
            RouteResult(type=TaskType.WEB_SEARCH, response="Found info", provider="serper"),
            RouteResult(type=TaskType.CODE, response="def foo(): pass", provider="anthropic"),
        ]
        synthesis = await planner.synthesize("Research and code", results)
        assert synthesis == "Here is the combined answer."

    @pytest.mark.asyncio
    async def test_economy_mode_prefers_free(self, pipeline):
        router = pipeline["router"]
        subtask = Subtask(type=TaskType.GENERAL, content="hello", complexity=Complexity.LOW)
        result = await router.route(subtask, mode=Mode.ECONOMY)
        # Should NOT use anthropic (paid) for a low-complexity general task in economy mode
        assert result.provider in ("groq", "gemini")
