"""Tests for search providers (Serper, Tavily)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from smartsplit.config import ProviderConfig
from smartsplit.exceptions import ProviderError
from smartsplit.providers.serper import SerperProvider
from smartsplit.providers.tavily import TavilyProvider


def _test_config(**overrides: object) -> ProviderConfig:
    defaults = {"api_key": "test-key", "enabled": True}
    defaults.update(overrides)
    return ProviderConfig(**defaults)


# ── Serper ────────────────────────────────────────────────────


class TestSerperProvider:
    @pytest.fixture
    def provider(self):
        mock_http = MagicMock()
        return SerperProvider(config=_test_config(), http=mock_http)

    @pytest.mark.asyncio
    async def test_search_with_results(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "organic": [
                {"title": "Result 1", "snippet": "Snippet 1", "link": "https://example.com/1"},
                {"title": "Result 2", "snippet": "Snippet 2", "link": "https://example.com/2"},
            ]
        }
        provider.http.post = AsyncMock(return_value=response)

        result = await provider.search("test query")
        assert "Result 1" in result
        assert "Snippet 1" in result
        assert "example.com" in result

    @pytest.mark.asyncio
    async def test_search_with_answer_box(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "answerBox": {"answer": "42"},
            "organic": [],
        }
        provider.http.post = AsyncMock(return_value=response)

        result = await provider.search("meaning of life")
        assert "42" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"organic": []}
        provider.http.post = AsyncMock(return_value=response)

        result = await provider.search("impossible query")
        assert result == "No results found."

    @pytest.mark.asyncio
    async def test_search_sends_correct_headers(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"organic": []}
        provider.http.post = AsyncMock(return_value=response)

        await provider.search("test")
        call_args = provider.http.post.call_args
        assert call_args[1]["headers"]["X-API-KEY"] == "test-key"

    @pytest.mark.asyncio
    async def test_search_invalid_json_response(self, provider):
        """response.json() raising ValueError should produce a ProviderError."""
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.side_effect = ValueError("Expecting value")
        provider.http.post = AsyncMock(return_value=response)

        with pytest.raises(ProviderError, match="Invalid JSON"):
            await provider.search("test")


# ── Tavily ────────────────────────────────────────────────────


class TestTavilyProvider:
    @pytest.fixture
    def provider(self):
        mock_http = MagicMock()
        return TavilyProvider(config=_test_config(), http=mock_http)

    @pytest.mark.asyncio
    async def test_search_with_results(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "results": [
                {"title": "Result 1", "content": "Content 1", "url": "https://example.com/1"},
            ]
        }
        provider.http.post = AsyncMock(return_value=response)

        result = await provider.search("test query")
        assert "Result 1" in result
        assert "Content 1" in result

    @pytest.mark.asyncio
    async def test_search_with_answer(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "answer": "The answer is 42",
            "results": [],
        }
        provider.http.post = AsyncMock(return_value=response)

        result = await provider.search("what is the answer")
        assert "42" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"results": []}
        provider.http.post = AsyncMock(return_value=response)

        result = await provider.search("impossible query")
        assert result == "No results found."

    @pytest.mark.asyncio
    async def test_search_sends_api_key_in_body(self, provider):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"results": []}
        provider.http.post = AsyncMock(return_value=response)

        await provider.search("test")
        call_args = provider.http.post.call_args
        assert call_args[1]["json"]["api_key"] == "test-key"

    @pytest.mark.asyncio
    async def test_search_invalid_json_response(self, provider):
        """response.json() raising ValueError should produce a ProviderError."""
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.side_effect = ValueError("Expecting value")
        provider.http.post = AsyncMock(return_value=response)

        with pytest.raises(ProviderError, match="Invalid JSON"):
            await provider.search("test")
