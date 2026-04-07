"""OpenRouter provider — 29 free models including Qwen3 Coder 480B."""

from __future__ import annotations

from smartsplit.providers.base import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"
    api_url = "https://openrouter.ai/api/v1/chat/completions"
