"""Tests for SmartSplit provider registry."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from smartsplit.config import ProviderConfig
from smartsplit.exceptions import NoProviderAvailableError, ProviderError
from smartsplit.models import ContextTier, TokenUsage
from smartsplit.providers.anthropic_adapter import (
    anthropic_to_openai as _convert_from_anthropic,
)
from smartsplit.providers.anthropic_adapter import (
    openai_to_anthropic as _convert_to_anthropic,
)
from smartsplit.providers.base import LLMProvider, SearchProvider
from smartsplit.providers.registry import (
    _CONTEXT_TIER_MAX_CHARS,
    ProviderRegistry,
    _failure_weight,
    _sanitize_error,
)

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

    def test_brain_name_not_available_logs_warning(self, make_config):
        """Brain name set to a provider that is not in the registry (line 411)."""
        config = make_config(["groq"])
        registry = ProviderRegistry(
            config.providers,
            httpx.AsyncClient(),
            brain_name="anthropic",
        )
        # Brain is set but not in providers — should still initialize
        assert registry.brain_name == "anthropic"
        assert registry.get("anthropic") is None

    def test_brain_name_in_providers(self, make_config):
        """Brain name set to a provider that IS in the registry."""
        config = make_config(["groq", "anthropic"])
        registry = ProviderRegistry(
            config.providers,
            httpx.AsyncClient(),
            brain_name="anthropic",
        )
        assert registry.brain_name == "anthropic"
        assert registry.get("anthropic") is not None

    def test_no_providers_configured(self):
        """No providers configured at all — warning logged."""
        registry = ProviderRegistry({}, httpx.AsyncClient())
        assert len(registry.get_all()) == 0


# ── Free LLM fallback ───────────────────────────────────────


class TestCallFreeLLM:
    @pytest.mark.asyncio
    async def test_prefers_specified(self, make_registry):
        registry = make_registry(["groq", "gemini"])
        registry.get("gemini").complete = AsyncMock(return_value=("gemini ok", TokenUsage()))
        result = await registry.call_free_llm("test", prefer="gemini")
        assert result == "gemini ok"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, make_registry):
        registry = make_registry(["groq", "gemini"])
        registry.get("groq").complete = AsyncMock(side_effect=Exception("down"))
        registry.get("gemini").complete = AsyncMock(return_value=("gemini fallback", TokenUsage()))
        result = await registry.call_free_llm("test", prefer="groq")
        assert result == "gemini fallback"

    @pytest.mark.asyncio
    async def test_all_fail_raises(self, make_registry):
        registry = make_registry(["groq"])
        registry.get("groq").complete = AsyncMock(side_effect=Exception("down"))
        with pytest.raises(NoProviderAvailableError):
            await registry.call_free_llm("test")

    @pytest.mark.asyncio
    async def test_brain_as_last_resort_fallback(self, make_config):
        """Brain is added as last-resort fallback when not in free priority (line 546)."""
        config = make_config(["groq", "anthropic"])
        registry = ProviderRegistry(
            config.providers,
            httpx.AsyncClient(),
            brain_name="anthropic",
        )
        # Groq fails
        registry.get("groq").complete = AsyncMock(side_effect=Exception("down"))
        # Anthropic (brain) succeeds as last-resort
        registry.get("anthropic").complete = AsyncMock(return_value=("brain fallback", TokenUsage()))
        result = await registry.call_free_llm("test", prefer="groq")
        assert result == "brain fallback"

    @pytest.mark.asyncio
    async def test_timeout_records_failure(self, make_registry):
        """Timeout in call_free_llm records circuit breaker failure (lines 568-569)."""
        registry = make_registry(["groq", "gemini"])

        async def slow_complete(prompt, **kwargs):
            raise TimeoutError("timed out")

        registry.get("groq").complete = slow_complete
        registry.get("gemini").complete = AsyncMock(return_value=("gemini ok", TokenUsage()))
        result = await registry.call_free_llm("test", prefer="groq")
        assert result == "gemini ok"
        # Groq should have a failure recorded
        assert len(registry.circuit_breaker._failures.get("groq", [])) > 0 or not registry.circuit_breaker.is_healthy(
            "groq"
        )


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
        for _ in range(5):
            registry.circuit_breaker.record_failure("groq")
        assert registry.circuit_breaker.is_healthy("groq") is False

    def test_recovery_enters_half_open(self, make_registry):
        cb = make_registry(["groq"]).circuit_breaker
        for _ in range(5):
            cb.record_failure("groq")
        assert cb.is_healthy("groq") is False
        # Backdate cooldown
        cb._open_until["groq"] = time.time() - 1
        # First call enters half-open — returns True (probe allowed)
        assert cb.is_healthy("groq") is True
        assert "groq" in cb._half_open
        # Second call while probe in flight — blocked
        assert cb.is_healthy("groq") is False
        # Probe succeeds → fully closed
        cb.record_success("groq")
        assert "groq" not in cb._half_open
        assert cb.is_healthy("groq") is True

    def test_half_open_probe_failure_reopens(self, make_registry):
        cb = make_registry(["groq"]).circuit_breaker
        for _ in range(5):
            cb.record_failure("groq")
        cb._open_until["groq"] = time.time() - 1
        assert cb.is_healthy("groq") is True  # enters half-open
        # Probe fails → immediate reopen, no need for 5 failures
        cb.record_failure("groq")
        assert "groq" not in cb._half_open
        assert cb.is_healthy("groq") is False
        assert cb._consecutive_trips["groq"] == 2

    def test_success_resets_failures(self, make_registry):
        registry = make_registry(["groq"])
        registry.circuit_breaker.record_failure("groq")
        registry.circuit_breaker.record_failure("groq")
        registry.circuit_breaker.record_success("groq")
        registry.circuit_breaker.record_failure("groq")
        # Should not be open — success reset the failure history
        assert registry.circuit_breaker.is_healthy("groq") is True

    def test_exponential_backoff(self, make_registry):
        cb = make_registry(["groq"]).circuit_breaker
        # First trip: 60s cooldown
        for _ in range(5):
            cb.record_failure("groq")
        assert cb.is_healthy("groq") is False
        assert cb._consecutive_trips["groq"] == 1
        # Recover via half-open probe success, then trip again
        cb._open_until["groq"] = time.time() - 1
        assert cb.is_healthy("groq") is True  # half-open
        cb.record_success("groq")  # probe OK → closed
        for _ in range(5):
            cb.record_failure("groq")
        assert cb._consecutive_trips["groq"] == 2

    def test_graduated_reset_needs_multiple_successes(self, make_registry):
        cb = make_registry(["groq"]).circuit_breaker
        # Trip once
        for _ in range(5):
            cb.record_failure("groq")
        assert cb._consecutive_trips["groq"] == 1
        # Recover via half-open
        cb._open_until["groq"] = time.time() - 1
        cb.is_healthy("groq")  # half-open
        cb.record_success("groq")  # 1st success
        assert cb._consecutive_trips.get("groq") == 1  # not reset yet
        cb.record_success("groq")  # 2nd success
        assert cb._consecutive_trips.get("groq") == 1  # still not reset
        cb.record_success("groq")  # 3rd success → reset
        assert cb._consecutive_trips.get("groq") is None

    def test_auth_error_trips_immediately(self, make_registry):
        cb = make_registry(["groq"]).circuit_breaker
        exc = httpx.HTTPStatusError(
            "",
            request=httpx.Request("POST", "https://api.example.com"),
            response=httpx.Response(401),
        )
        # Single auth error should trip (weight = threshold)
        cb.record_failure("groq", exc)
        assert cb.is_healthy("groq") is False

    def test_rate_limit_half_weight(self, make_registry):
        cb = make_registry(["groq"]).circuit_breaker
        exc = httpx.HTTPStatusError(
            "",
            request=httpx.Request("POST", "https://api.example.com"),
            response=httpx.Response(429),
        )
        # 9 rate limits = 4.5 points < 5 threshold → still healthy
        for _ in range(9):
            cb.record_failure("groq", exc)
        assert cb.is_healthy("groq") is True
        # 10th pushes to 5.0 → trips
        cb.record_failure("groq", exc)
        assert cb.is_healthy("groq") is False

    def test_get_unhealthy_includes_half_open(self, make_registry):
        cb = make_registry(["groq", "gemini"]).circuit_breaker
        for _ in range(5):
            cb.record_failure("groq")
        unhealthy = cb.get_unhealthy()
        assert "groq" in unhealthy
        assert "gemini" not in unhealthy
        # Half-open providers are also reported as unhealthy
        cb._open_until["groq"] = time.time() - 1
        cb.is_healthy("groq")  # enters half-open
        assert "groq" in cb.get_unhealthy()

    @pytest.mark.asyncio
    async def test_call_free_llm_skips_unhealthy(self, make_registry):
        registry = make_registry(["groq", "gemini"])
        # Open circuit for groq
        for _ in range(5):
            registry.circuit_breaker.record_failure("groq")
        # Mock gemini to succeed
        registry.get("gemini").complete = AsyncMock(return_value=("gemini ok", TokenUsage()))
        result = await registry.call_free_llm("test", prefer="groq")
        assert result == "gemini ok"


# ── Context Tiers ──────────────────────────────────────────────


def _make_tier_registry(
    providers: dict[str, ContextTier],
) -> ProviderRegistry:
    """Build a registry with specific context tiers per provider."""
    from smartsplit.config import DEFAULT_FREE_LLM_PRIORITY

    configs: dict[str, ProviderConfig] = {}
    models = {"groq": "llama-3.3-70b-versatile", "gemini": "gemini-2.5-flash"}
    for name, tier in providers.items():
        configs[name] = ProviderConfig(
            api_key="test_key",
            enabled=True,
            model=models.get(name, "test-model"),
            context_tier=tier,
        )
    return ProviderRegistry(configs, httpx.AsyncClient(), list(DEFAULT_FREE_LLM_PRIORITY))


class TestContextTiers:
    """Test that call_free_llm truncates prompts based on provider context tier."""

    @pytest.mark.asyncio
    async def test_long_prompt_truncated_for_small_tier(self):
        """A prompt exceeding SMALL tier (4000 chars) should be truncated."""
        registry = _make_tier_registry({"groq": ContextTier.SMALL})
        provider = registry.get("groq")

        received_prompts: list[str] = []

        async def capture_prompt(prompt, **kwargs):
            received_prompts.append(prompt)
            return ("ok", TokenUsage())

        provider.complete = capture_prompt

        long_prompt = "x" * 5000
        await registry.call_free_llm(long_prompt, prefer="groq")

        assert len(received_prompts) == 1
        assert len(received_prompts[0]) == _CONTEXT_TIER_MAX_CHARS[ContextTier.SMALL]
        assert len(received_prompts[0]) == 4000

    @pytest.mark.asyncio
    async def test_short_prompt_not_truncated(self):
        """A prompt shorter than the tier limit should be passed unchanged."""
        registry = _make_tier_registry({"groq": ContextTier.SMALL})
        provider = registry.get("groq")

        received_prompts: list[str] = []

        async def capture_prompt(prompt, **kwargs):
            received_prompts.append(prompt)
            return ("ok", TokenUsage())

        provider.complete = capture_prompt

        short_prompt = "x" * 3000
        await registry.call_free_llm(short_prompt, prefer="groq")

        assert len(received_prompts) == 1
        assert received_prompts[0] == short_prompt

    @pytest.mark.asyncio
    async def test_fallback_provider_gets_more_context(self):
        """When small-tier provider fails, large-tier fallback receives more context."""
        registry = _make_tier_registry(
            {"groq": ContextTier.SMALL, "gemini": ContextTier.LARGE},
        )
        groq = registry.get("groq")
        gemini = registry.get("gemini")

        received_by_groq: list[str] = []
        received_by_gemini: list[str] = []

        async def groq_fail(prompt, **kwargs):
            received_by_groq.append(prompt)
            raise RuntimeError("groq down")

        async def gemini_ok(prompt, **kwargs):
            received_by_gemini.append(prompt)
            return ("gemini ok", TokenUsage())

        groq.complete = groq_fail
        gemini.complete = gemini_ok

        long_prompt = "x" * 10_000
        result = await registry.call_free_llm(long_prompt, prefer="groq")

        assert result == "gemini ok"
        # Groq received truncated to SMALL tier
        assert len(received_by_groq[0]) == _CONTEXT_TIER_MAX_CHARS[ContextTier.SMALL]
        # Gemini received full prompt (within LARGE tier limit of 64000)
        assert len(received_by_gemini[0]) == 10_000


# ── Helpers (_sanitize_error, _failure_weight) ─────────────────


class TestHelpers:
    def test_sanitize_error_redacts_api_keys(self):
        msg = "Error with key sk-ant-abcdef1234567890abcdef1234567890 in request"
        sanitized = _sanitize_error(Exception(msg))
        assert "sk-ant-" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_sanitize_error_redacts_groq_key(self):
        msg = "Auth failed: gsk_abcdef1234567890abcdef1234567890abcdef"
        sanitized = _sanitize_error(Exception(msg))
        assert "gsk_" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_sanitize_error_no_key_unchanged(self):
        msg = "Connection refused"
        sanitized = _sanitize_error(Exception(msg))
        assert sanitized == "Connection refused"

    def test_failure_weight_none_returns_1(self):
        assert _failure_weight(None) == 1.0

    def test_failure_weight_auth_error(self):
        exc = httpx.HTTPStatusError(
            "",
            request=httpx.Request("POST", "https://api.example.com"),
            response=httpx.Response(401),
        )
        assert _failure_weight(exc) == 5.0

    def test_failure_weight_forbidden_error(self):
        exc = httpx.HTTPStatusError(
            "",
            request=httpx.Request("POST", "https://api.example.com"),
            response=httpx.Response(403),
        )
        assert _failure_weight(exc) == 5.0

    def test_failure_weight_rate_limit(self):
        exc = httpx.HTTPStatusError(
            "",
            request=httpx.Request("POST", "https://api.example.com"),
            response=httpx.Response(429),
        )
        assert _failure_weight(exc) == 0.5

    def test_failure_weight_server_error(self):
        exc = httpx.HTTPStatusError(
            "",
            request=httpx.Request("POST", "https://api.example.com"),
            response=httpx.Response(500),
        )
        assert _failure_weight(exc) == 1.0

    def test_failure_weight_generic_exception(self):
        assert _failure_weight(RuntimeError("boom")) == 1.0


# ── Anthropic conversion ───────────────────────────────────────


class TestConvertToAnthropic:
    """Tests for _convert_to_anthropic (lines 42-140)."""

    def test_basic_user_message(self):
        body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "ignored",
        }
        result = _convert_to_anthropic(body, "claude-3-opus")
        assert result["model"] == "claude-3-opus"
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["max_tokens"] == 4096  # default

    def test_system_message_extracted(self):
        body = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = _convert_to_anthropic(body, "claude-3-opus")
        assert result["system"] == "You are helpful."
        # System message should NOT appear in messages
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_system_message_structured_content(self):
        body = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Part 1"},
                        "Part 2",
                    ],
                },
                {"role": "user", "content": "Hi"},
            ],
        }
        result = _convert_to_anthropic(body, "claude-3-opus")
        assert "Part 1" in result["system"]
        assert "Part 2" in result["system"]

    def test_multiple_system_messages_joined(self):
        body = {
            "messages": [
                {"role": "system", "content": "First system"},
                {"role": "system", "content": "Second system"},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = _convert_to_anthropic(body, "claude-3-opus")
        assert "First system" in result["system"]
        assert "Second system" in result["system"]

    def test_assistant_tool_calls_converted(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Let me search",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query": "test"}',
                            },
                        }
                    ],
                }
            ],
        }
        result = _convert_to_anthropic(body, "claude-3-opus")
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 2
        assert msg["content"][0] == {"type": "text", "text": "Let me search"}
        assert msg["content"][1]["type"] == "tool_use"
        assert msg["content"][1]["id"] == "call_123"
        assert msg["content"][1]["name"] == "web_search"
        assert msg["content"][1]["input"] == {"query": "test"}

    def test_tool_message_converted_to_tool_result(self):
        body = {
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "Search results here",
                }
            ],
        }
        result = _convert_to_anthropic(body, "claude-3-opus")
        msg = result["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "call_123"
        assert msg["content"][0]["content"] == "Search results here"

    def test_tools_converted(self):
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    },
                }
            ],
        }
        result = _convert_to_anthropic(body, "claude-3-opus")
        assert "tools" in result
        assert result["tools"][0]["name"] == "web_search"
        assert result["tools"][0]["description"] == "Search the web"
        assert "input_schema" in result["tools"][0]

    def test_tool_choice_auto(self):
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": "auto",
        }
        result = _convert_to_anthropic(body, "claude-3-opus")
        assert result["tool_choice"] == {"type": "auto"}

    def test_tool_choice_required(self):
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": "required",
        }
        result = _convert_to_anthropic(body, "claude-3-opus")
        assert result["tool_choice"] == {"type": "any"}

    def test_tool_choice_none_removes_tools(self):
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "test", "description": "", "parameters": {}}}],
            "tool_choice": "none",
        }
        result = _convert_to_anthropic(body, "claude-3-opus")
        assert "tools" not in result

    def test_tool_choice_specific_function(self):
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"function": {"name": "web_search"}},
        }
        result = _convert_to_anthropic(body, "claude-3-opus")
        assert result["tool_choice"] == {"type": "tool", "name": "web_search"}

    def test_temperature_passed_through(self):
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
        }
        result = _convert_to_anthropic(body, "claude-3-opus")
        assert result["temperature"] == 0.7

    def test_max_tokens_from_body(self):
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8192,
        }
        result = _convert_to_anthropic(body, "claude-3-opus")
        assert result["max_tokens"] == 8192

    def test_assistant_tool_calls_with_dict_arguments(self):
        """Arguments already parsed as dict (not string)."""
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_456",
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "/tmp/test.txt"},
                            },
                        }
                    ],
                }
            ],
        }
        result = _convert_to_anthropic(body, "claude-3-opus")
        msg = result["messages"][0]
        # No text block when content is None/falsy
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "tool_use"
        assert msg["content"][0]["input"] == {"path": "/tmp/test.txt"}


class TestConvertFromAnthropic:
    """Tests for _convert_from_anthropic (lines 148-192)."""

    def test_text_response(self):
        data = {
            "id": "msg_123",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = _convert_from_anthropic(data, "claude-3-opus")
        assert result["id"] == "chatcmpl-msg_123"
        assert result["object"] == "chat.completion"
        assert result["model"] == "claude-3-opus"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_tool_use_response(self):
        data = {
            "id": "msg_456",
            "content": [
                {"type": "text", "text": "Searching..."},
                {
                    "type": "tool_use",
                    "id": "tu_789",
                    "name": "web_search",
                    "input": {"query": "test"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }
        result = _convert_from_anthropic(data, "claude-3-opus")
        msg = result["choices"][0]["message"]
        assert msg["content"] == "Searching..."
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "tu_789"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "web_search"
        assert json.loads(tc["function"]["arguments"]) == {"query": "test"}
        assert tc["index"] == 0
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_max_tokens_finish_reason(self):
        data = {
            "id": "msg_max",
            "content": [{"type": "text", "text": "Truncated"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 5, "output_tokens": 100},
        }
        result = _convert_from_anthropic(data, "claude-3-opus")
        assert result["choices"][0]["finish_reason"] == "length"

    def test_no_text_content(self):
        """Tool-only response with no text blocks."""
        data = {
            "id": "msg_tool_only",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_001",
                    "name": "read_file",
                    "input": {"path": "/tmp/test"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {},
        }
        result = _convert_from_anthropic(data, "claude-3-opus")
        assert result["choices"][0]["message"]["content"] is None
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1

    def test_empty_usage(self):
        data = {
            "id": "msg_empty",
            "content": [{"type": "text", "text": "Hi"}],
            "stop_reason": "end_turn",
        }
        result = _convert_from_anthropic(data, "claude-3-opus")
        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["completion_tokens"] == 0
        assert result["usage"]["total_tokens"] == 0

    def test_multiple_tool_use_blocks(self):
        data = {
            "id": "msg_multi",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "a.py"}},
                {"type": "tool_use", "id": "tu_2", "name": "grep", "input": {"pattern": "foo"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        result = _convert_from_anthropic(data, "claude-3-opus")
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["index"] == 0
        assert tool_calls[1]["index"] == 1


# ── call_brain ──────────────────────────────────────────────────


class TestCallBrain:
    """Tests for ProviderRegistry.call_brain (lines 433-471)."""

    @pytest.mark.asyncio
    async def test_call_brain_success(self, make_config):
        config = make_config(["groq"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient(), brain_name="groq")
        registry.get("groq").complete = AsyncMock(return_value=("brain says hi", TokenUsage()))
        result, usage = await registry.call_brain("test prompt")
        assert result == "brain says hi"

    @pytest.mark.asyncio
    async def test_call_brain_fallback_on_failure(self, make_config):
        """Brain fails, falls back to next provider in priority (line 462)."""
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient(), brain_name="groq")
        registry.get("groq").complete = AsyncMock(side_effect=Exception("brain down"))
        registry.get("gemini").complete = AsyncMock(return_value=("gemini fallback", TokenUsage()))
        result, usage = await registry.call_brain("test")
        assert result == "gemini fallback"

    @pytest.mark.asyncio
    async def test_call_brain_timeout(self, make_config):
        """Brain times out, falls back to next (lines 465-466)."""
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient(), brain_name="groq")

        async def slow_complete(prompt, **kwargs):
            raise TimeoutError("timed out")

        registry.get("groq").complete = slow_complete
        registry.get("gemini").complete = AsyncMock(return_value=("gemini ok", TokenUsage()))
        result, usage = await registry.call_brain("test")
        assert result == "gemini ok"

    @pytest.mark.asyncio
    async def test_call_brain_all_fail_raises(self, make_config):
        config = make_config(["groq"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient(), brain_name="groq")
        registry.get("groq").complete = AsyncMock(side_effect=Exception("down"))
        with pytest.raises(NoProviderAvailableError, match="brain"):
            await registry.call_brain("test")

    @pytest.mark.asyncio
    async def test_call_brain_skips_unhealthy(self, make_config):
        """Circuit breaker open for brain, falls back (lines 451, 453-454)."""
        config = make_config(["groq", "gemini"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient(), brain_name="groq")
        # Trip circuit for groq
        for _ in range(5):
            registry.circuit_breaker.record_failure("groq")
        registry.get("gemini").complete = AsyncMock(return_value=("gemini ok", TokenUsage()))
        result, usage = await registry.call_brain("test")
        assert result == "gemini ok"

    @pytest.mark.asyncio
    async def test_call_brain_with_messages(self, make_config):
        config = make_config(["groq"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient(), brain_name="groq")
        registry.get("groq").complete = AsyncMock(return_value=("response", TokenUsage()))
        msgs = [{"role": "user", "content": "hello"}]
        result, usage = await registry.call_brain("test", messages=msgs)
        assert result == "response"
        registry.get("groq").complete.assert_called_once_with("test", messages=msgs)

    @pytest.mark.asyncio
    async def test_call_brain_no_brain_name(self, make_config):
        """No brain_name set — uses _BRAIN_PRIORITY order."""
        config = make_config(["groq"])
        registry = ProviderRegistry(config.providers, httpx.AsyncClient(), brain_name="")
        registry.get("groq").complete = AsyncMock(return_value=("groq ok", TokenUsage()))
        result, usage = await registry.call_brain("test")
        assert result == "groq ok"


# ── proxy_to_brain ──────────────────────────────────────────────


class TestProxyToBrain:
    """Tests for ProviderRegistry.proxy_to_brain (lines 480-538)."""

    @pytest.mark.asyncio
    async def test_proxy_to_brain_openai_compatible(self, make_config):
        """Proxy to a standard OpenAI-compatible provider (groq)."""
        config = make_config(["groq"])
        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_http.post = AsyncMock(return_value=mock_response)

        registry = ProviderRegistry(config.providers, mock_http, brain_name="groq")
        body = {
            "messages": [{"role": "user", "content": "test"}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
            "stream": True,
        }
        result = await registry.proxy_to_brain(body)
        assert result["choices"][0]["message"]["content"] == "hello"
        # Verify stream was removed
        call_kwargs = mock_http.post.call_args
        sent_body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "stream" not in sent_body

    @pytest.mark.asyncio
    async def test_proxy_to_brain_anthropic(self, make_config):
        """Proxy to Anthropic — converts to/from Anthropic format."""
        config = make_config(["anthropic"])
        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "id": "msg_123",
            "content": [{"type": "text", "text": "Anthropic response"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_http.post = AsyncMock(return_value=mock_response)

        registry = ProviderRegistry(config.providers, mock_http, brain_name="anthropic")
        body = {
            "messages": [{"role": "user", "content": "test"}],
        }
        result = await registry.proxy_to_brain(body)
        # Should be converted back to OpenAI format
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Anthropic response"
        # Verify Anthropic URL was used
        call_args = mock_http.post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert url.startswith("https://api.anthropic.com/")

    @pytest.mark.asyncio
    async def test_proxy_to_brain_gemini(self, make_config):
        """Proxy to Gemini uses OpenAI-compatible endpoint."""
        config = make_config(["gemini"])
        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "gemini response"}}],
        }
        mock_http.post = AsyncMock(return_value=mock_response)

        registry = ProviderRegistry(config.providers, mock_http, brain_name="gemini")
        body = {"messages": [{"role": "user", "content": "test"}]}
        result = await registry.proxy_to_brain(body)
        assert result["choices"][0]["message"]["content"] == "gemini response"
        call_args = mock_http.post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert url.startswith("https://generativelanguage.googleapis.com/")

    @pytest.mark.asyncio
    async def test_proxy_to_brain_timeout_fallback(self, make_config):
        """Timeout on first brain candidate, fallback to next."""
        config = make_config(["groq", "gemini"])
        call_count = 0

        async def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timed out")
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "fallback"}}]}
            return mock_resp

        mock_http = MagicMock()
        mock_http.post = mock_post

        registry = ProviderRegistry(config.providers, mock_http, brain_name="groq")
        body = {"messages": [{"role": "user", "content": "test"}]}
        result = await registry.proxy_to_brain(body)
        assert result["choices"][0]["message"]["content"] == "fallback"

    @pytest.mark.asyncio
    async def test_proxy_to_brain_error_fallback(self, make_config):
        """HTTP error on first brain candidate, fallback to next."""
        config = make_config(["groq", "gemini"])
        call_count = 0

        async def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPStatusError(
                    "Server Error",
                    request=httpx.Request("POST", url),
                    response=httpx.Response(500),
                )
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "fallback ok"}}]}
            return mock_resp

        mock_http = MagicMock()
        mock_http.post = mock_post

        registry = ProviderRegistry(config.providers, mock_http, brain_name="groq")
        body = {"messages": [{"role": "user", "content": "test"}]}
        result = await registry.proxy_to_brain(body)
        assert result["choices"][0]["message"]["content"] == "fallback ok"

    @pytest.mark.asyncio
    async def test_proxy_to_brain_all_fail_raises(self, make_config):
        """All brain candidates fail — raises NoProviderAvailableError."""
        config = make_config(["groq"])
        mock_http = MagicMock()
        mock_http.post = AsyncMock(side_effect=Exception("down"))

        registry = ProviderRegistry(config.providers, mock_http, brain_name="groq")
        body = {"messages": [{"role": "user", "content": "test"}]}
        with pytest.raises(NoProviderAvailableError, match="brain"):
            await registry.proxy_to_brain(body)

    @pytest.mark.asyncio
    async def test_proxy_to_brain_skips_unhealthy(self, make_config):
        """Circuit breaker open for brain — skips to next."""
        config = make_config(["groq", "gemini"])
        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "gemini ok"}}]}
        mock_http.post = AsyncMock(return_value=mock_response)

        registry = ProviderRegistry(config.providers, mock_http, brain_name="groq")
        # Trip circuit for groq
        for _ in range(5):
            registry.circuit_breaker.record_failure("groq")
        body = {"messages": [{"role": "user", "content": "test"}]}
        result = await registry.proxy_to_brain(body)
        assert result["choices"][0]["message"]["content"] == "gemini ok"
