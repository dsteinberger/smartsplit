"""Tests for custom-format providers (Anthropic, Gemini) and OpenAI-compatible backups."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from smartsplit.config import ProviderConfig
from smartsplit.exceptions import ProviderError
from smartsplit.providers.anthropic import AnthropicProvider
from smartsplit.providers.cloudflare import CloudflareProvider
from smartsplit.providers.gemini import GeminiProvider
from smartsplit.providers.huggingface import HuggingFaceProvider
from smartsplit.providers.perplexity import PerplexityProvider


def _test_config(**overrides: object) -> ProviderConfig:
    defaults = {"api_key": "test-key", "model": "test-model", "enabled": True}
    defaults.update(overrides)
    return ProviderConfig(**defaults)


# ── Anthropic ─────────────────────────────────────────────────


class TestAnthropicProvider:
    @pytest.fixture
    def provider(self):
        mock_http = MagicMock()
        return AnthropicProvider(config=_test_config(), http=mock_http)

    @pytest.mark.asyncio
    async def test_complete_success(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "content": [{"text": "Hello from Claude"}],
        }
        provider.http.post = AsyncMock(return_value=response)

        result, usage = await provider.complete("Say hello")
        assert result == "Hello from Claude"

    @pytest.mark.asyncio
    async def test_complete_with_messages(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"content": [{"text": "Done"}]}
        provider.http.post = AsyncMock(return_value=response)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Do something"},
        ]
        await provider.complete("Do something", messages=messages)

        # Verify system is separated and user/assistant are passed
        call_args = provider.http.post.call_args
        body = call_args[1]["json"]
        assert body["system"] == "You are helpful."
        assert len(body["messages"]) == 3  # user, assistant, user (no system)
        assert body["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_complete_with_model_override(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"content": [{"text": "Done"}]}
        provider.http.post = AsyncMock(return_value=response)

        await provider.complete("test", model="claude-opus-4")
        body = provider.http.post.call_args[1]["json"]
        assert body["model"] == "claude-opus-4"

    @pytest.mark.asyncio
    async def test_complete_http_error(self, provider):
        response = MagicMock()
        response.status_code = 402
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "insufficient", request=MagicMock(), response=response
        )
        provider.http.post = AsyncMock(return_value=response)

        with pytest.raises(ProviderError, match="HTTP 402"):
            await provider.complete("test")

    @pytest.mark.asyncio
    async def test_complete_timeout(self, provider):
        provider.http.post = AsyncMock(side_effect=httpx.ReadTimeout("timeout"))

        with pytest.raises(ProviderError, match="timed out"):
            await provider.complete("test")

    @pytest.mark.asyncio
    async def test_complete_invalid_json_response(self, provider):
        """response.json() raising ValueError should produce a ProviderError."""
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.side_effect = ValueError("Expecting value")
        provider.http.post = AsyncMock(return_value=response)

        with pytest.raises(ProviderError, match="Invalid JSON"):
            await provider.complete("test")


# ── Gemini ────────────────────────────────────────────────────


class TestGeminiProvider:
    @pytest.fixture
    def provider(self):
        mock_http = MagicMock()
        return GeminiProvider(config=_test_config(), http=mock_http)

    @pytest.mark.asyncio
    async def test_complete_success(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Hello from Gemini"}]}}],
        }
        provider.http.post = AsyncMock(return_value=response)

        result, usage = await provider.complete("Say hello")
        assert result == "Hello from Gemini"

    @pytest.mark.asyncio
    async def test_complete_with_messages(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Done"}]}}],
        }
        provider.http.post = AsyncMock(return_value=response)

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        await provider.complete("Hello", messages=messages)

        body = provider.http.post.call_args[1]["json"]
        # Gemini merges consecutive same-role messages (system+user → user)
        assert body["contents"][0]["role"] == "user"
        assert "Be helpful" in body["contents"][0]["parts"][0]["text"]
        assert "Hello" in body["contents"][0]["parts"][0]["text"]
        assert body["contents"][1]["role"] == "model"

    @pytest.mark.asyncio
    async def test_complete_with_model_override(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Done"}]}}],
        }
        provider.http.post = AsyncMock(return_value=response)

        await provider.complete("test", model="gemini-pro")
        url = provider.http.post.call_args[0][0]
        assert "gemini-pro" in url

    @pytest.mark.asyncio
    async def test_complete_http_error(self, provider):
        response = MagicMock()
        response.status_code = 429
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "rate limited", request=MagicMock(), response=response
        )
        provider.http.post = AsyncMock(return_value=response)

        with pytest.raises(ProviderError, match="HTTP 429"):
            await provider.complete("test")

    @pytest.mark.asyncio
    async def test_complete_timeout(self, provider):
        provider.http.post = AsyncMock(side_effect=httpx.ReadTimeout("timeout"))

        with pytest.raises(ProviderError, match="timed out"):
            await provider.complete("test")

    @pytest.mark.asyncio
    async def test_complete_invalid_json_response(self, provider):
        """response.json() raising ValueError should produce a ProviderError."""
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.side_effect = ValueError("Expecting value")
        provider.http.post = AsyncMock(return_value=response)

        with pytest.raises(ProviderError, match="Invalid JSON"):
            await provider.complete("test")


# ── Perplexity (OpenAI-compatible, Sonar models) ─────────────


class TestPerplexityProvider:
    @pytest.fixture
    def provider(self):
        return PerplexityProvider(config=_test_config(model="sonar"), http=MagicMock())

    def test_endpoint(self, provider):
        assert provider.api_url == "https://api.perplexity.ai/chat/completions"

    @pytest.mark.asyncio
    async def test_complete_success(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": "Sonar says hi"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
        provider.http.post = AsyncMock(return_value=response)
        result, usage = await provider.complete("ping")
        assert result == "Sonar says hi"
        assert usage.prompt_tokens == 3 and usage.completion_tokens == 4

    @pytest.mark.asyncio
    async def test_complete_http_error(self, provider):
        response = MagicMock()
        response.status_code = 429
        response.raise_for_status.side_effect = httpx.HTTPStatusError("rate", request=MagicMock(), response=response)
        provider.http.post = AsyncMock(return_value=response)
        with pytest.raises(ProviderError, match="HTTP 429"):
            await provider.complete("test")

    @pytest.mark.asyncio
    async def test_proxy_openai_forwards_body(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        provider.http.post = AsyncMock(return_value=response)
        result = await provider.proxy_openai_request({"model": "ignored", "messages": [], "stream": True})
        assert result["choices"][0]["message"]["content"] == "ok"
        body = provider.http.post.call_args[1]["json"]
        assert body["model"] == "sonar"  # overridden from config
        assert "stream" not in body  # stripped


# ── HuggingFace Router (OpenAI-compatible) ───────────────────


class TestHuggingFaceProvider:
    @pytest.fixture
    def provider(self):
        return HuggingFaceProvider(
            config=_test_config(model="Qwen/Qwen2.5-Coder-32B-Instruct"),
            http=MagicMock(),
        )

    def test_endpoint(self, provider):
        assert provider.api_url == "https://router.huggingface.co/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_complete_success(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": "HF response"}}]}
        provider.http.post = AsyncMock(return_value=response)
        result, _ = await provider.complete("ping")
        assert result == "HF response"

    @pytest.mark.asyncio
    async def test_bearer_auth_header(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": "x"}}]}
        provider.http.post = AsyncMock(return_value=response)
        await provider.complete("ping")
        headers = provider.http.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-key"


# ── Cloudflare Workers AI ────────────────────────────────────


class TestCloudflareProvider:
    _VALID_ACCOUNT = "a" * 32

    def test_requires_account_id(self, monkeypatch):
        monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
        with pytest.raises(ProviderError, match="CLOUDFLARE_ACCOUNT_ID not set"):
            CloudflareProvider(config=_test_config(), http=MagicMock())

    def test_rejects_malformed_account_id(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "not-hex")
        with pytest.raises(ProviderError, match="32-char hex"):
            CloudflareProvider(config=_test_config(), http=MagicMock())

    def test_builds_api_url(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", self._VALID_ACCOUNT)
        provider = CloudflareProvider(config=_test_config(), http=MagicMock())
        assert provider.api_url == (
            f"https://api.cloudflare.com/client/v4/accounts/{self._VALID_ACCOUNT}/ai/v1/chat/completions"
        )

    @pytest.mark.asyncio
    async def test_complete_success(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", self._VALID_ACCOUNT)
        provider = CloudflareProvider(
            config=_test_config(model="@cf/meta/llama-3"),
            http=MagicMock(),
        )
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": "CF reply"}}]}
        provider.http.post = AsyncMock(return_value=response)
        result, _ = await provider.complete("ping")
        assert result == "CF reply"
