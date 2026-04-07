"""OpenAI provider — paid, strong general-purpose model."""

from __future__ import annotations

from smartsplit.providers.base import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"
    api_url = "https://api.openai.com/v1/chat/completions"
    model = "gpt-4o-mini"
