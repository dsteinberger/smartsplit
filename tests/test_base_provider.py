"""Tests for base provider classes and _extract helper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from smartsplit.exceptions import ProviderError
from smartsplit.providers.base import OpenAICompatibleProvider


class TestExtract:
    """Test the _extract() safe nested accessor."""

    def _make_provider(self):
        p = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
        p.name = "test"
        return p

    def test_extract_simple(self):
        p = self._make_provider()
        data = {"choices": [{"message": {"content": "hello"}}]}
        assert p._extract(data, "choices", 0, "message", "content") == "hello"

    def test_extract_missing_key(self):
        p = self._make_provider()
        with pytest.raises(ProviderError, match="Unexpected API response"):
            p._extract({"foo": "bar"}, "choices", 0)

    def test_extract_index_out_of_range(self):
        p = self._make_provider()
        with pytest.raises(ProviderError, match="Unexpected API response"):
            p._extract({"choices": []}, "choices", 0)

    def test_extract_non_string_result(self):
        p = self._make_provider()
        with pytest.raises(ProviderError, match="Expected string"):
            p._extract({"choices": [{"val": 42}]}, "choices", 0, "val")

    def test_extract_deep_nesting(self):
        p = self._make_provider()
        data = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        assert p._extract(data, "a", "b", "c", "d", "e") == "deep"


class TestOpenAICompatibleProvider:
    """Test the OpenAI-compatible provider base class."""

    @pytest.fixture
    def provider(self):
        """Create a concrete subclass for testing."""

        class TestProvider(OpenAICompatibleProvider):
            name = "test"
            api_url = "https://api.test.com/v1/chat/completions"
            model = "test-model"

        mock_http = MagicMock()
        return TestProvider(api_key="test-key", http=mock_http)

    @pytest.mark.asyncio
    async def test_complete_success(self, provider):
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": "Hello world"}}],
        }
        provider.http.post = AsyncMock(return_value=response)

        result = await provider.complete("Say hello")
        assert result == "Hello world"

        # Verify request format
        call_args = provider.http.post.call_args
        assert call_args[0][0] == "https://api.test.com/v1/chat/completions"
        body = call_args[1]["json"]
        assert body["model"] == "test-model"
        assert body["messages"][0]["content"] == "Say hello"

    @pytest.mark.asyncio
    async def test_complete_http_error(self, provider):
        import httpx

        response = MagicMock()
        response.status_code = 429
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "rate limited", request=MagicMock(), response=response
        )
        provider.http.post = AsyncMock(return_value=response)

        with pytest.raises(httpx.HTTPStatusError):
            await provider.complete("test")

    @pytest.mark.asyncio
    async def test_complete_malformed_json(self, provider):
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.return_value = {"unexpected": "format"}
        provider.http.post = AsyncMock(return_value=response)

        with pytest.raises(ProviderError):
            await provider.complete("test")

    @pytest.mark.asyncio
    async def test_complete_empty_choices(self, provider):
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.return_value = {"choices": []}
        provider.http.post = AsyncMock(return_value=response)

        with pytest.raises(ProviderError):
            await provider.complete("test")
