"""OpenRouter provider — 29 free models including Qwen3 Coder 480B."""

from __future__ import annotations

from smartsplit.providers.base import LLMProvider


class OpenRouterProvider(LLMProvider):
    name = "openrouter"

    async def complete(self, prompt: str) -> str:
        response = await self.http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": "qwen/qwen3-coder-480b:free",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 4096,
            },
        )
        response.raise_for_status()
        return self._extract(response.json(), "choices", 0, "message", "content")
