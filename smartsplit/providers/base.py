"""Abstract base classes for providers (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from smartsplit.exceptions import ProviderError


class BaseProvider(ABC):
    """Base class for all providers."""

    name: str

    def __init__(self, api_key: str, http: httpx.AsyncClient) -> None:
        self.api_key = api_key
        self.http = http

    def _extract(self, data: object, *keys: str | int) -> str:
        """Safely extract a nested value from a parsed API response.

        Raises ProviderError with a clear message if the path doesn't exist.
        """
        current = data
        path_so_far: list[str] = []
        for key in keys:
            path_so_far.append(str(key))
            try:
                current = current[key]
            except (KeyError, IndexError, TypeError) as exc:
                raise ProviderError(
                    self.name,
                    f"Unexpected API response structure at {'.'.join(path_so_far)}",
                ) from exc
        if not isinstance(current, str):
            raise ProviderError(
                self.name,
                f"Expected string at {'.'.join(path_so_far)}, got {type(current).__name__}",
            )
        return current


class LLMProvider(BaseProvider):
    """A provider that generates text from a prompt."""

    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """Send a prompt and return the completion text."""


class OpenAICompatibleProvider(LLMProvider):
    """Base for providers using the OpenAI chat completions API format.

    Subclasses only need to set ``name``, ``api_url``, and ``model``.
    """

    api_url: str
    model: str

    async def complete(self, prompt: str) -> str:
        response = await self.http.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 4096,
            },
        )
        response.raise_for_status()
        return self._extract(response.json(), "choices", 0, "message", "content")


class SearchProvider(BaseProvider):
    """A provider that returns web search results."""

    @abstractmethod
    async def search(self, query: str) -> str:
        """Execute a search query and return formatted results."""
