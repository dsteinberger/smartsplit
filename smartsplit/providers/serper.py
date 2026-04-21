"""Serper provider — Google Search API, free tier."""

from __future__ import annotations

import httpx

from smartsplit.exceptions import ProviderError
from smartsplit.providers.base import SearchProvider, http_error_to_provider_error


class SerperProvider(SearchProvider):
    name = "serper"

    async def search(self, query: str) -> str:
        n = self.config.max_search_results
        try:
            response = await self.http.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self.api_key},
                json={"q": query, "num": n},
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
        if data.get("answerBox"):
            ab = data["answerBox"]
            results.append(f"**Answer:** {ab.get('answer', ab.get('snippet', ''))}")

        for item in data.get("organic", [])[:n]:
            results.append(f"**{item.get('title', '')}**\n{item.get('snippet', '')}\n{item.get('link', '')}")

        return "\n\n".join(results) if results else "No results found."
