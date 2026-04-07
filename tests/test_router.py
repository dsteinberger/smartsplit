"""Tests for SmartSplit router scoring and routing logic."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import httpx
import pytest

from smartsplit.models import Complexity, Mode, ProviderType, RouteResult, Subtask, TaskType, TerminationState
from smartsplit.providers.registry import ProviderRegistry
from smartsplit.quota import QuotaTracker
from smartsplit.router import Router

# ── Scoring (additive weighted) ──────────────────────────────


class TestScoring:
    @pytest.mark.parametrize(
        "mode, free_wins",
        [
            pytest.param(Mode.ECONOMY, True, id="economy-favors-free"),
            pytest.param(Mode.QUALITY, False, id="quality-favors-paid"),
        ],
    )
    def test_mode_affects_free_vs_paid(self, make_config, tmp_path, mode, free_wins):
        config = make_config(["groq", "anthropic"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        subtask = Subtask(type=TaskType.CODE, content="test", complexity=Complexity.HIGH)
        score_free = router.score("groq", ProviderType.FREE, subtask, mode)
        score_paid = router.score("anthropic", ProviderType.PAID, subtask, mode)

        if free_wins:
            assert score_free > score_paid
        else:
            assert score_paid > score_free

    def test_high_complexity_boosts_quality_weight(self, make_config, tmp_path):
        """High complexity shifts weight toward quality → paid provider with high competence benefits."""
        config = make_config(["anthropic"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        low = Subtask(type=TaskType.CODE, content="test", complexity=Complexity.LOW)
        high = Subtask(type=TaskType.CODE, content="test", complexity=Complexity.HIGH)
        # Anthropic has competence 10/10 for code, so quality boost helps it
        score_low = router.score("anthropic", ProviderType.PAID, low, Mode.BALANCED)
        score_high = router.score("anthropic", ProviderType.PAID, high, Mode.BALANCED)
        assert score_high > score_low

    def test_availability_only_penalizes_below_threshold(self, make_config, tmp_path):
        """Availability should not penalize until below the danger threshold (10%)."""
        config = make_config(["groq"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        subtask = Subtask(type=TaskType.SUMMARIZE, content="test", complexity=Complexity.LOW)

        # 50% availability = well above threshold → should score same as 100%
        score_full = router.score("groq", ProviderType.FREE, subtask, Mode.BALANCED)
        quota._usage["groq"] = {"count": 7200, "last_reset": time.time(), "by_type": {}}
        score_half = router.score("groq", ProviderType.FREE, subtask, Mode.BALANCED)
        assert score_full == score_half  # both above threshold

    def test_availability_penalizes_below_threshold(self, make_config, tmp_path):
        """Near-exhausted providers should score lower."""
        config = make_config(["groq"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        subtask = Subtask(type=TaskType.SUMMARIZE, content="test", complexity=Complexity.LOW)
        score_full = router.score("groq", ProviderType.FREE, subtask, Mode.BALANCED)

        # 99% used → below danger threshold
        quota._usage["groq"] = {"count": 14200, "last_reset": time.time(), "by_type": {}}
        score_danger = router.score("groq", ProviderType.FREE, subtask, Mode.BALANCED)
        assert score_full > score_danger

    def test_score_is_always_positive(self, make_config, tmp_path):
        config = make_config(["groq"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        subtask = Subtask(type=TaskType.GENERAL, content="test", complexity=Complexity.MEDIUM)
        score = router.score("groq", ProviderType.FREE, subtask, Mode.BALANCED)
        assert score > 0

    def test_unknown_task_type_uses_general(self, make_config, tmp_path):
        config = make_config(["groq"])
        config.competence_table.pop("summarize", None)
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        subtask = Subtask(type=TaskType.SUMMARIZE, content="test", complexity=Complexity.LOW)
        score = router.score("groq", ProviderType.FREE, subtask, Mode.BALANCED)
        assert score > 0


# ── Routing ──────────────────────────────────────────────────


class TestRouting:
    @pytest.mark.asyncio
    async def test_web_search_routes_to_search_provider(self, make_config, tmp_path):
        config = make_config(["groq", "serper"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        registry.get("serper").search = AsyncMock(return_value="search results")

        subtask = Subtask(type=TaskType.WEB_SEARCH, content="test", complexity=Complexity.LOW)
        result = await router.route(subtask)
        assert isinstance(result, RouteResult)
        assert result.provider == "serper"

    @pytest.mark.asyncio
    async def test_llm_task_skips_search_providers(self, make_config, tmp_path):
        config = make_config(["groq", "serper"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        registry.get("groq").complete = AsyncMock(return_value="llm response")

        subtask = Subtask(type=TaskType.SUMMARIZE, content="test", complexity=Complexity.LOW)
        result = await router.route(subtask)
        assert result.provider == "groq"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, make_config, tmp_path):
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        registry.get("groq").complete = AsyncMock(side_effect=Exception("down"))
        registry.get("gemini").complete = AsyncMock(return_value="gemini ok")

        subtask = Subtask(type=TaskType.SUMMARIZE, content="test", complexity=Complexity.LOW)
        result = await router.route(subtask)
        assert result.response == "gemini ok"

    @pytest.mark.asyncio
    async def test_no_provider_available(self, make_config, tmp_path):
        config = make_config([])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        subtask = Subtask(type=TaskType.CODE, content="test", complexity=Complexity.LOW)
        result = await router.route(subtask)
        assert result.provider == "none"

    @pytest.mark.asyncio
    async def test_all_providers_fail(self, make_config, tmp_path):
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        registry.get("groq").complete = AsyncMock(side_effect=Exception("down"))
        registry.get("gemini").complete = AsyncMock(side_effect=Exception("down"))

        subtask = Subtask(type=TaskType.SUMMARIZE, content="test", complexity=Complexity.LOW)
        result = await router.route(subtask)
        assert result.provider == "none"
        assert "All providers failed" in result.response

    @pytest.mark.asyncio
    async def test_mode_override(self, make_config, tmp_path):
        config = make_config(["groq", "anthropic"], mode=Mode.ECONOMY)
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        registry.get("groq").complete = AsyncMock(
            return_value="A decent groq response that is long enough to pass the gate"
        )
        registry.get("anthropic").complete = AsyncMock(
            return_value="A detailed Claude response with complex algorithm implementation and explanation"
        )

        subtask = Subtask(type=TaskType.CODE, content="test", complexity=Complexity.HIGH)
        result = await router.route(subtask, mode=Mode.QUALITY)
        assert result.provider == "anthropic"


# ── Quality Gates ───────────────────────────────────────────


class TestQualityGates:
    def test_empty_response_fails_gate(self):
        subtask = Subtask(type=TaskType.CODE, content="test", complexity=Complexity.LOW)
        assert not Router.passes_quality_gate("", subtask)

    def test_short_response_fails_gate(self):
        subtask = Subtask(type=TaskType.CODE, content="test", complexity=Complexity.HIGH)
        assert not Router.passes_quality_gate("ok", subtask)

    def test_error_pattern_fails_gate(self):
        subtask = Subtask(type=TaskType.CODE, content="test", complexity=Complexity.LOW)
        assert not Router.passes_quality_gate("I cannot help with that request.", subtask)

    def test_good_response_passes_gate(self):
        subtask = Subtask(type=TaskType.CODE, content="test", complexity=Complexity.HIGH)
        response = "Here is the implementation of the sorting algorithm with O(n log n) complexity."
        assert Router.passes_quality_gate(response, subtask)

    def test_rate_limit_fails_gate(self):
        subtask = Subtask(type=TaskType.GENERAL, content="test", complexity=Complexity.LOW)
        assert not Router.passes_quality_gate("Error: rate limit exceeded, try again later.", subtask)

    def test_legitimate_error_content_passes_gate(self):
        """Content discussing errors should NOT be flagged as low quality."""
        subtask = Subtask(type=TaskType.CODE, content="explain errors", complexity=Complexity.HIGH)
        response = (
            "The IndexError exception occurs when you try to access an index "
            "that is out of range. Here is how to handle it properly with "
            "try/except blocks and validation checks."
        )
        assert Router.passes_quality_gate(response, subtask)

    @pytest.mark.asyncio
    async def test_quality_gate_escalation_in_balanced_mode(self, make_config, tmp_path):
        """In balanced mode, a low-quality response should trigger escalation to next provider."""
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        # groq returns garbage, gemini returns a good response
        registry.get("groq").complete = AsyncMock(return_value="I cannot help with that.")
        registry.get("gemini").complete = AsyncMock(
            return_value="Here is a detailed and helpful answer to your question about this topic."
        )

        subtask = Subtask(type=TaskType.GENERAL, content="explain X", complexity=Complexity.MEDIUM)
        result = await router.route(subtask, mode=Mode.BALANCED)
        assert result.provider == "gemini"

    @pytest.mark.asyncio
    async def test_quality_gate_skipped_in_economy_mode(self, make_config, tmp_path):
        """Economy mode should skip quality gates and accept the first response."""
        config = make_config(["groq"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        registry.get("groq").complete = AsyncMock(return_value="I cannot help with that.")

        subtask = Subtask(type=TaskType.GENERAL, content="test", complexity=Complexity.LOW)
        result = await router.route(subtask, mode=Mode.ECONOMY)
        # Economy mode accepts the response even though it would fail the gate
        assert result.provider == "groq"

    @pytest.mark.asyncio
    async def test_quality_gate_returns_last_response_if_all_fail(self, make_config, tmp_path):
        """If all providers fail the quality gate, return the first provider's response."""
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        registry.get("groq").complete = AsyncMock(return_value="I cannot do this.")
        registry.get("gemini").complete = AsyncMock(return_value="As an AI, I'm unable to help.")

        subtask = Subtask(type=TaskType.GENERAL, content="test", complexity=Complexity.MEDIUM)
        result = await router.route(subtask, mode=Mode.BALANCED)
        # Should return a response (not "All providers failed") since providers DID respond
        assert result.provider != "none"


# ── Structured Traces ───────────────────────────────────────


class TestStructuredTraces:
    @pytest.mark.asyncio
    async def test_successful_route_has_completed_termination(self, make_config, tmp_path):
        config = make_config(["groq"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        registry.get("groq").complete = AsyncMock(return_value="A good response that passes quality check.")

        subtask = Subtask(type=TaskType.GENERAL, content="test prompt", complexity=Complexity.LOW)
        result = await router.route(subtask)
        assert result.termination == TerminationState.COMPLETED
        assert result.escalations == []
        assert result.estimated_tokens > 0

    @pytest.mark.asyncio
    async def test_escalation_records_quality_gate_reason(self, make_config, tmp_path):
        """Gemini scores higher than Groq for general tasks. Make gemini fail the gate,
        so it escalates to groq."""
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        # Gemini (higher score) fails gate, groq (lower score) passes
        registry.get("gemini").complete = AsyncMock(return_value="I cannot help with that.")
        registry.get("groq").complete = AsyncMock(
            return_value="Here is a detailed and helpful answer to your question about this topic."
        )

        subtask = Subtask(type=TaskType.GENERAL, content="explain X", complexity=Complexity.MEDIUM)
        result = await router.route(subtask, mode=Mode.BALANCED)
        assert result.termination == TerminationState.ESCALATED
        assert len(result.escalations) >= 1
        assert result.escalations[0].reason == "quality_gate"
        assert result.escalations[0].from_provider == "gemini"
        assert result.escalations[0].to_provider == "groq"

    @pytest.mark.asyncio
    async def test_no_provider_termination(self, make_config, tmp_path):
        config = make_config([])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        subtask = Subtask(type=TaskType.CODE, content="test", complexity=Complexity.LOW)
        result = await router.route(subtask)
        assert result.termination == TerminationState.NO_PROVIDER

    @pytest.mark.asyncio
    async def test_all_failed_termination(self, make_config, tmp_path):
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        registry.get("groq").complete = AsyncMock(side_effect=Exception("down"))
        registry.get("gemini").complete = AsyncMock(side_effect=Exception("down"))

        subtask = Subtask(type=TaskType.GENERAL, content="test", complexity=Complexity.LOW)
        result = await router.route(subtask)
        assert result.termination == TerminationState.ALL_FAILED

    @pytest.mark.asyncio
    async def test_quality_gate_fallback_termination(self, make_config, tmp_path):
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        registry.get("groq").complete = AsyncMock(return_value="I cannot do this.")
        registry.get("gemini").complete = AsyncMock(return_value="As an AI, I'm unable to help.")

        subtask = Subtask(type=TaskType.GENERAL, content="test", complexity=Complexity.MEDIUM)
        result = await router.route(subtask, mode=Mode.QUALITY)
        assert result.termination == TerminationState.QUALITY_GATE_FALLBACK

    @pytest.mark.asyncio
    async def test_escalation_records_provider_error(self, make_config, tmp_path):
        """Gemini scores higher. Make gemini crash, groq should catch it with an escalation record."""
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        # Gemini (higher score) crashes, groq (lower score) works
        registry.get("gemini").complete = AsyncMock(side_effect=Exception("down"))
        registry.get("groq").complete = AsyncMock(return_value="Groq works fine with a good enough response here.")

        subtask = Subtask(type=TaskType.GENERAL, content="test prompt", complexity=Complexity.LOW)
        result = await router.route(subtask)
        assert result.provider == "groq"
        assert any(e.reason == "provider_error" for e in result.escalations)
        assert result.escalations[0].from_provider == "gemini"
        assert result.escalations[0].to_provider == "groq"
