"""Google Gemini provider — free tier, strong for summaries, reasoning, and code."""

from __future__ import annotations

from smartsplit.providers.base import LLMProvider


class GeminiProvider(LLMProvider):
    name = "gemini"

    async def complete(self, prompt: str) -> str:
        response = await self.http.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            headers={"x-goog-api-key": self.api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096},
            },
        )
        response.raise_for_status()
        return self._extract(response.json(), "candidates", 0, "content", "parts", 0, "text")
