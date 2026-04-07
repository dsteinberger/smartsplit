"""Tests for SmartSplit provider registry."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import httpx
import pytest

from smartsplit.config import ProviderConfig
from smartsplit.exceptions import NoProviderAvailableError, ProviderError
from smartsplit.providers.base import LLMProvider, SearchProvider

# ── Registry initialization ──────────────────────────────────


class TestRegistryInit:
    def test_creates_enabled_providers(self, make_registry):
        registry = make_registry(["groq", "gemini"])
        providers = registry.get_all()
        assert "groq" in providers
        assert "gemini" in providers

    def test_skips_disabled(self):
        from smartsplit.providers.registry import ProviderRegistry

        configs = {"groq": ProviderConfig(api_key="k", enabled=False)}
        registry = ProviderRegistry(configs, httpx.AsyncClient())
        assert registry.get("groq") is None

    def test_skips_no_key(self):
        from smartsplit.providers.registry import ProviderRegistry

        configs = {"groq": ProviderConfig(api_key="", enabled=True)}
        registry = ProviderRegistry(configs, httpx.AsyncClient())
        assert registry.get("groq") is None

    @pytest.mark.parametrize(
        "provider, expected_type",
        [
            pytest.param("groq", LLMProvider, id="groq-is-llm"),
            pytest.param("cerebras", LLMProvider, id="cerebras-is-llm"),
            pytest.param("gemini", LLMProvider, id="gemini-is-llm"),
            pytest.param("deepseek", LLMProvider, id="deepseek-is-llm"),
            pytest.param("openrouter", LLMProvider, id="openrouter-is-llm"),
            pytest.param("mistral", LLMProvider, id="mistral-is-llm"),
            pytest.param("anthropic", LLMProvider, id="anthropic-is-llm"),
            pytest.param("openai", LLMProvider, id="openai-is-llm"),
            pytest.param("serper", SearchProvider, id="serper-is-search"),
            pytest.param("tavily", SearchProvider, id="tavily-is-search"),
        ],
    )
    def test_provider_types(self, make_registry, provider, expected_type):
        registry = make_registry([provider])
        assert isinstance(registry.get(provider), expected_type)

    def test_typed_lookups(self, make_registry):
        registry = make_registry(["groq", "serper"])
        assert "groq" in registry.get_llm_providers()
        assert "groq" not in registry.get_search_providers()
        assert "serper" in registry.get_search_providers()
        assert "serper" not in registry.get_llm_providers()

    def test_unknown_provider_skipped(self):
        from smartsplit.providers.registry import ProviderRegistry

        configs = {"unknown_ai": ProviderConfig(api_key="k", enabled=True)}
        registry = ProviderRegistry(configs, httpx.AsyncClient())
        assert registry.get("unknown_ai") is None


# ── Free LLM fallback ───────────────────────────────────────


class TestCallFreeLLM:
    @pytest.mark.asyncio
    async def test_prefers_specified(self, make_registry):
        registry = make_registry(["groq", "gemini"])
        registry.get("gemini").complete = AsyncMock(return_value="gemini ok")
        result = await registry.call_free_llm("test", prefer="gemini")
        assert result == "gemini ok"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, make_registry):
        registry = make_registry(["groq", "gemini"])
        registry.get("groq").complete = AsyncMock(side_effect=Exception("down"))
        registry.get("gemini").complete = AsyncMock(return_value="gemini fallback")
        result = await registry.call_free_llm("test", prefer="groq")
        assert result == "gemini fallback"

    @pytest.mark.asyncio
    async def test_all_fail_raises(self, make_registry):
        registry = make_registry(["groq"])
        registry.get("groq").complete = AsyncMock(side_effect=Exception("down"))
        with pytest.raises(NoProviderAvailableError):
            await registry.call_free_llm("test")


# ── Direct calls ─────────────────────────────────────────────


class TestDirectCalls:
    @pytest.mark.asyncio
    async def test_call_llm_unknown_raises(self, make_registry):
        registry = make_registry([])
        with pytest.raises(ProviderError, match="nonexistent"):
            await registry.call_llm("nonexistent", "test")

    @pytest.mark.asyncio
    async def test_call_search_unknown_raises(self, make_registry):
        registry = make_registry([])
        with pytest.raises(ProviderError, match="nonexistent"):
            await registry.call_search("nonexistent", "test")

    @pytest.mark.asyncio
    async def test_call_search_on_llm_provider_raises(self, make_registry):
        registry = make_registry(["groq"])
        with pytest.raises(ProviderError):
            await registry.call_search("groq", "test")

    @pytest.mark.asyncio
    async def test_call_llm_on_search_provider_raises(self, make_registry):
        registry = make_registry(["serper"])
        with pytest.raises(ProviderError):
            await registry.call_llm("serper", "test")


# ── Circuit Breaker ─────────────────────────────────────────


class TestCircuitBreaker:
    def test_healthy_by_default(self, make_registry):
        registry = make_registry(["groq"])
        assert registry.circuit_breaker.is_healthy("groq") is True

    def test_stays_healthy_below_threshold(self, make_registry):
        registry = make_registry(["groq"])
        registry.circuit_breaker.record_failure("groq")
        registry.circuit_breaker.record_failure("groq")
        assert registry.circuit_breaker.is_healthy("groq") is True

    def test_opens_after_threshold(self, make_registry):
        registry = make_registry(["groq"])
        for _ in range(3):
            registry.circuit_breaker.record_failure("groq")
        assert registry.circuit_breaker.is_healthy("groq") is False

    def test_recovery_after_timeout(self, make_registry):
        registry = make_registry(["groq"])
        for _ in range(3):
            registry.circuit_breaker.record_failure("groq")
        assert registry.circuit_breaker.is_healthy("groq") is False
        # Fake recovery by backdating
        registry.circuit_breaker._open_until["groq"] = time.time() - 1
        assert registry.circuit_breaker.is_healthy("groq") is True

    def test_success_resets_failures(self, make_registry):
        registry = make_registry(["groq"])
        registry.circuit_breaker.record_failure("groq")
        registry.circuit_breaker.record_failure("groq")
        registry.circuit_breaker.record_success("groq")
        registry.circuit_breaker.record_failure("groq")
        # Should not be open — success reset the counter
        assert registry.circuit_breaker.is_healthy("groq") is True

    def test_get_unhealthy_list(self, make_registry):
        registry = make_registry(["groq", "gemini"])
        for _ in range(3):
            registry.circuit_breaker.record_failure("groq")
        unhealthy = registry.circuit_breaker.get_unhealthy()
        assert "groq" in unhealthy
        assert "gemini" not in unhealthy

    @pytest.mark.asyncio
    async def test_call_free_llm_skips_unhealthy(self, make_registry):
        registry = make_registry(["groq", "gemini"])
        # Open circuit for groq
        for _ in range(3):
            registry.circuit_breaker.record_failure("groq")
        # Mock gemini to succeed
        registry.get("gemini").complete = AsyncMock(return_value="gemini ok")
        result = await registry.call_free_llm("test", prefer="groq")
        assert result == "gemini ok"
