"""Tavily provider — AI-optimized search, free tier."""

from __future__ import annotations

import httpx

from smartsplit.exceptions import ProviderError
from smartsplit.providers.base import SearchProvider, http_error_to_provider_error


class TavilyProvider(SearchProvider):
    name = "tavily"

    async def search(self, query: str) -> str:
        n = self.config.max_search_results
        try:
            response = await self.http.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": n,
                    "search_depth": "basic",
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise http_error_to_provider_error(self.name, e) from e
        except httpx.TimeoutException as e:
            raise ProviderError(self.name, "Request timed out") from e
        try:
            data = response.json()
        except ValueError as e:
            raise ProviderError(self.name, "Invalid JSON in API response") from e

        results: list[str] = []
        if data.get("answer"):
            results.append(f"**Answer:** {data['answer']}")

        for item in data.get("results", [])[:n]:
            results.append(f"**{item.get('title', '')}**\n{item.get('content', '')[:200]}\n{item.get('url', '')}")

        return "\n\n".join(results) if results else "No results found."
