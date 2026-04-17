"""Tests for SmartSplit router scoring and routing logic."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import httpx
import pytest

from smartsplit.exceptions import ProviderError
from smartsplit.models import (
    Complexity,
    Mode,
    ProviderType,
    RouteResult,
    Subtask,
    TaskType,
    TerminationState,
    TokenUsage,
)
from smartsplit.providers.registry import ProviderRegistry
from smartsplit.routing.quota import QuotaTracker
from smartsplit.routing.router import Router

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
        registry.get("groq").complete = AsyncMock(return_value=("llm response", TokenUsage()))

        subtask = Subtask(type=TaskType.SUMMARIZE, content="test", complexity=Complexity.LOW)
        result = await router.route(subtask)
        assert result.provider == "groq"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, make_config, tmp_path):
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        registry.get("groq").complete = AsyncMock(side_effect=ProviderError("test", "down"))
        registry.get("gemini").complete = AsyncMock(return_value=("gemini ok", TokenUsage()))

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
        registry.get("groq").complete = AsyncMock(side_effect=ProviderError("test", "down"))
        registry.get("gemini").complete = AsyncMock(side_effect=ProviderError("test", "down"))

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
            return_value=("A decent groq response that is long enough to pass the gate", TokenUsage())
        )
        registry.get("anthropic").complete = AsyncMock(
            return_value=(
                "A detailed Claude response with complex algorithm implementation and explanation",
                TokenUsage(),
            )
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

    def test_padded_refusal_beyond_scan_length_passes(self):
        """A refusal hidden beyond the 80-char scan window is NOT caught.

        This is a known trade-off: short scan avoids false positives on
        legitimate responses that start with 'I apologize...'. Padded
        refusals that start beyond 80 chars slip through.
        """
        subtask = Subtask(type=TaskType.GENERAL, content="test", complexity=Complexity.LOW)
        # Pad beyond 80 chars, then refusal pattern
        response = (
            "Thank you for the interesting question. Let me think about this carefully. "
            + "I cannot help with that request."
        )
        # The refusal starts at position ~76, but the padding pushes it beyond scan range
        assert Router.passes_quality_gate(response, subtask)

    def test_short_refusal_within_scan_length_fails(self):
        """A refusal within the 80-char scan window is correctly caught."""
        subtask = Subtask(type=TaskType.GENERAL, content="test", complexity=Complexity.LOW)
        response = "I'm sorry, but I cannot help with that particular request."
        assert not Router.passes_quality_gate(response, subtask)

    @pytest.mark.asyncio
    async def test_quality_gate_escalation_in_balanced_mode(self, make_config, tmp_path):
        """In balanced mode, a low-quality response should trigger escalation to next provider."""
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        # groq returns garbage, gemini returns a good response
        registry.get("groq").complete = AsyncMock(return_value=("I cannot help with that.", TokenUsage()))
        registry.get("gemini").complete = AsyncMock(
            return_value=("Here is a detailed and helpful answer to your question about this topic.", TokenUsage())
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
        registry.get("groq").complete = AsyncMock(return_value=("I cannot help with that.", TokenUsage()))

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
        registry.get("groq").complete = AsyncMock(return_value=("I cannot do this.", TokenUsage()))
        registry.get("gemini").complete = AsyncMock(return_value=("As an AI, I'm unable to help.", TokenUsage()))

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
        registry.get("groq").complete = AsyncMock(
            return_value=("A good response that passes quality check.", TokenUsage())
        )

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
        registry.get("gemini").complete = AsyncMock(return_value=("I cannot help with that.", TokenUsage()))
        registry.get("groq").complete = AsyncMock(
            return_value=("Here is a detailed and helpful answer to your question about this topic.", TokenUsage())
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
        registry.get("groq").complete = AsyncMock(side_effect=ProviderError("test", "down"))
        registry.get("gemini").complete = AsyncMock(side_effect=ProviderError("test", "down"))

        subtask = Subtask(type=TaskType.GENERAL, content="test", complexity=Complexity.LOW)
        result = await router.route(subtask)
        assert result.termination == TerminationState.ALL_FAILED

    @pytest.mark.asyncio
    async def test_quality_gate_fallback_termination(self, make_config, tmp_path):
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)
        registry.get("groq").complete = AsyncMock(return_value=("I cannot do this.", TokenUsage()))
        registry.get("gemini").complete = AsyncMock(return_value=("As an AI, I'm unable to help.", TokenUsage()))

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
        registry.get("gemini").complete = AsyncMock(side_effect=ProviderError("test", "down"))
        registry.get("groq").complete = AsyncMock(
            return_value=("Groq works fine with a good enough response here.", TokenUsage())
        )

        subtask = Subtask(type=TaskType.GENERAL, content="test prompt", complexity=Complexity.LOW)
        result = await router.route(subtask)
        assert result.provider == "groq"
        assert any(e.reason == "provider_error" for e in result.escalations)
        assert result.escalations[0].from_provider == "gemini"
        assert result.escalations[0].to_provider == "groq"


# ── Tier logic (fast/strong) ────────────────────────────────


class TestTierLogic:
    def test_free_provider_always_fast(self):
        from smartsplit.config import ProviderConfig
        from smartsplit.routing.router import _resolve_tier

        free = ProviderConfig(type=ProviderType.FREE)
        assert _resolve_tier(free, Complexity.LOW) == "fast"
        assert _resolve_tier(free, Complexity.MEDIUM) == "fast"
        assert _resolve_tier(free, Complexity.HIGH) == "fast"

    def test_paid_high_complexity_uses_strong(self):
        from smartsplit.config import ProviderConfig
        from smartsplit.routing.router import _resolve_tier

        paid = ProviderConfig(type=ProviderType.PAID, fast_model="haiku", strong_model="sonnet")
        assert _resolve_tier(paid, Complexity.HIGH) == "strong"
        assert _resolve_tier(paid, Complexity.HIGH, Mode.ECONOMY) == "strong"

    def test_paid_medium_uses_strong_in_quality_mode(self):
        from smartsplit.config import ProviderConfig
        from smartsplit.routing.router import _resolve_tier

        paid = ProviderConfig(type=ProviderType.PAID, fast_model="haiku", strong_model="sonnet")
        assert _resolve_tier(paid, Complexity.MEDIUM, Mode.QUALITY) == "strong"
        assert _resolve_tier(paid, Complexity.MEDIUM, Mode.BALANCED) == "fast"
        assert _resolve_tier(paid, Complexity.MEDIUM, Mode.ECONOMY) == "fast"

    def test_paid_low_always_fast(self):
        from smartsplit.config import ProviderConfig
        from smartsplit.routing.router import _resolve_tier

        paid = ProviderConfig(type=ProviderType.PAID, fast_model="haiku", strong_model="sonnet")
        assert _resolve_tier(paid, Complexity.LOW, Mode.QUALITY) == "fast"
        assert _resolve_tier(paid, Complexity.LOW, Mode.ECONOMY) == "fast"

    def test_no_strong_model_always_fast(self):
        from smartsplit.config import ProviderConfig
        from smartsplit.routing.router import _resolve_tier

        paid = ProviderConfig(type=ProviderType.PAID, fast_model="haiku", strong_model="")
        assert _resolve_tier(paid, Complexity.HIGH) == "fast"

    def test_get_model_for_tier(self):
        from smartsplit.config import ProviderConfig
        from smartsplit.routing.router import _get_model_for_tier

        paid = ProviderConfig(type=ProviderType.PAID, fast_model="haiku", strong_model="sonnet")
        assert _get_model_for_tier(paid, "fast") == "haiku"
        assert _get_model_for_tier(paid, "strong") == "sonnet"

    def test_get_model_free_returns_none(self):
        from smartsplit.config import ProviderConfig
        from smartsplit.routing.router import _get_model_for_tier

        free = ProviderConfig(type=ProviderType.FREE)
        assert _get_model_for_tier(free, "fast") is None
        assert _get_model_for_tier(free, "strong") is None


# ── Provider overrides ──────────────────────────────────────


class TestProviderOverrides:
    @pytest.mark.asyncio
    async def test_override_forces_provider(self, make_config, tmp_path):
        """Override should force the specified provider for the task type."""
        config = make_config(["groq", "anthropic"], overrides={"code": "anthropic"})
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        # Mock the LLM call
        provider = registry.get("anthropic")
        provider.complete = AsyncMock(return_value=("result from anthropic", TokenUsage()))

        subtask = Subtask(type=TaskType.CODE, content="write a function", complexity=Complexity.HIGH)
        result = await router.route(subtask)
        assert result.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_override_fallback_when_circuit_breaker_open(self, make_config, tmp_path):
        """If overridden provider is down, fallback to normal scoring."""
        config = make_config(["groq", "anthropic"], overrides={"code": "anthropic"})
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        # Open circuit breaker for anthropic
        registry.circuit_breaker._open_until["anthropic"] = time.time() + 9999

        # Mock groq (the fallback)
        provider = registry.get("groq")
        provider.complete = AsyncMock(return_value=("result from groq", TokenUsage()))

        subtask = Subtask(type=TaskType.CODE, content="write a function", complexity=Complexity.HIGH)
        result = await router.route(subtask)
        assert result.provider == "groq"

    @pytest.mark.asyncio
    async def test_override_ignored_when_provider_not_configured(self, make_config, tmp_path):
        """Override for a provider that doesn't exist should be ignored."""
        config = make_config(["groq"], overrides={"code": "anthropic"})
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        provider = registry.get("groq")
        provider.complete = AsyncMock(return_value=("result from groq", TokenUsage()))

        subtask = Subtask(type=TaskType.CODE, content="write a function", complexity=Complexity.HIGH)
        result = await router.route(subtask)
        assert result.provider == "groq"

    def test_no_overrides_by_default(self, make_config):
        """Config without overrides should have empty dict."""
        config = make_config(["groq"])
        assert config.overrides == {}


# ── Fast-model retry, context errors, refusal check ──────────


class TestFastModelRetryAndContextErrors:
    @pytest.mark.asyncio
    async def test_strong_fails_fast_succeeds(self, make_config, tmp_path):
        """Strong model throws ProviderError → router retries the fast model on the same provider."""
        config = make_config(["anthropic"], mode=Mode.QUALITY)
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        provider = registry.get("anthropic")
        strong_model = config.providers["anthropic"].strong_model or "claude-sonnet-4-6-20250514"
        provider.config.strong_model = strong_model
        provider.config.fast_model = "claude-haiku-fast"

        def _mock_complete(prompt, model=None, messages=None, **_kw):
            if model == strong_model:
                raise ProviderError("anthropic", "boom")
            return (
                "fast model rescue that is long enough to pass quality gate",
                TokenUsage(prompt_tokens=1, completion_tokens=2),
            )

        provider.complete = AsyncMock(side_effect=_mock_complete)
        subtask = Subtask(type=TaskType.CODE, content="write code", complexity=Complexity.HIGH)
        result = await router.route(subtask, mode=Mode.QUALITY)
        assert result.provider == "anthropic"
        assert "fast model rescue" in result.response

    @pytest.mark.asyncio
    async def test_strong_and_fast_both_fail(self, make_config, tmp_path):
        """Strong AND fast model fail → provider marked failed, circuit breaker records failure."""
        config = make_config(["anthropic", "groq"], mode=Mode.QUALITY)
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        anthropic = registry.get("anthropic")
        anthropic.config.strong_model = "claude-sonnet-4-6-20250514"
        anthropic.config.fast_model = "claude-haiku-fast"
        anthropic.complete = AsyncMock(side_effect=ProviderError("anthropic", "down"))

        groq = registry.get("groq")
        groq.complete = AsyncMock(
            return_value=("groq fallback response long enough to pass quality gate", TokenUsage())
        )

        subtask = Subtask(type=TaskType.CODE, content="write code", complexity=Complexity.HIGH)
        result = await router.route(subtask, mode=Mode.QUALITY)
        assert result.provider == "groq"

    @pytest.mark.asyncio
    async def test_context_too_long_does_not_penalize(self, make_config, tmp_path):
        """`context length exceeded` → try next provider, do NOT record circuit-breaker failure."""
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        # Gemini is scored higher for reasoning → tried first. Make it fail on context, groq succeeds.
        gemini = registry.get("gemini")
        gemini.complete = AsyncMock(side_effect=ProviderError("gemini", "context length exceeded"))
        groq = registry.get("groq")
        long_response = "A proper reasoning answer " + ("x" * 400)
        groq.complete = AsyncMock(return_value=(long_response, TokenUsage()))

        subtask = Subtask(type=TaskType.REASONING, content="big prompt", complexity=Complexity.LOW)
        result = await router.route(subtask)
        assert result.provider == "groq"
        # Context error should NOT trip the breaker for gemini
        assert registry.circuit_breaker.is_healthy("gemini")
        assert any(e.reason == "context_too_long" for e in result.escalations)

    @pytest.mark.asyncio
    async def test_refusal_check_escalates(self, make_config, tmp_path):
        """LLM refusal detected on a mid-length response → escalate to next provider."""
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        # Response in the 50-300 char window that passes pattern check but is a refusal
        ambiguous = "Ok, here is something: actually I am not going to complete that particular task"
        registry.get("groq").complete = AsyncMock(return_value=(ambiguous, TokenUsage()))
        registry.get("gemini").complete = AsyncMock(
            return_value=("A proper answer that clearly passes the quality gate and is long enough", TokenUsage())
        )
        # Force LLM refusal check to return True for groq's response
        registry.call_free_llm = AsyncMock(return_value="yes")

        subtask = Subtask(type=TaskType.REASONING, content="explain recursion", complexity=Complexity.MEDIUM)
        result = await router.route(subtask)
        assert result.provider == "gemini"
        assert any(e.reason == "llm_refusal_check" for e in result.escalations)

    @pytest.mark.asyncio
    async def test_messages_are_rewritten_with_enriched_content(self, make_config, tmp_path):
        """When a subtask carries messages, the router overwrites the last user message with subtask.content."""
        config = make_config(["groq"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient())
        quota = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        router = Router(registry, quota, config)

        # Response > 300 chars so the refusal-check LLM path is skipped (otherwise
        # call_args would capture the refusal-check call, not the routing call).
        long_response = "A detailed reasoning answer " + ("x" * 400)
        mock = AsyncMock(return_value=(long_response, TokenUsage()))
        registry.get("groq").complete = mock

        subtask = Subtask(
            type=TaskType.REASONING,
            content="ENRICHED content",
            complexity=Complexity.LOW,
            messages=[{"role": "user", "content": "original user prompt"}],
        )
        await router.route(subtask)
        assert mock.call_args.kwargs["messages"][-1] == {"role": "user", "content": "ENRICHED content"}
