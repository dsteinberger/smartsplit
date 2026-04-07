"""Serper provider — Google Search API, free tier."""

from __future__ import annotations

from smartsplit.providers.base import SearchProvider


class SerperProvider(SearchProvider):
    name = "serper"

    async def search(self, query: str) -> str:
        response = await self.http.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": self.api_key},
            json={"q": query, "num": 5},
        )
        response.raise_for_status()
        data = response.json()

        results: list[str] = []
        if data.get("answerBox"):
            ab = data["answerBox"]
            results.append(f"**Answer:** {ab.get('answer', ab.get('snippet', ''))}")

        for item in data.get("organic", [])[:5]:
            results.append(f"**{item.get('title', '')}**\n{item.get('snippet', '')}\n{item.get('link', '')}")

        return "\n\n".join(results) if results else "No results found."
