"""Anthropic Claude provider — paid, best for code and reasoning."""

from __future__ import annotations

from smartsplit.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    async def complete(self, prompt: str) -> str:
        response = await self.http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6-20250514",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        return self._extract(response.json(), "content", 0, "text")
