"""Tavily provider — AI-optimized search, free tier."""

from __future__ import annotations

from smartsplit.providers.base import SearchProvider


class TavilyProvider(SearchProvider):
    name = "tavily"

    async def search(self, query: str) -> str:
        response = await self.http.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": 5,
                "search_depth": "basic",
            },
        )
        response.raise_for_status()
        data = response.json()

        results: list[str] = []
        if data.get("answer"):
            results.append(f"**Answer:** {data['answer']}")

        for item in data.get("results", [])[:5]:
            results.append(f"**{item.get('title', '')}**\n{item.get('content', '')[:200]}\n{item.get('url', '')}")

        return "\n\n".join(results) if results else "No results found."
