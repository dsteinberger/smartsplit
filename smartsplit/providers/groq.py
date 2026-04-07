"""Groq provider — free, fast inference via Llama models."""

from __future__ import annotations

from smartsplit.providers.base import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    api_url = "https://api.groq.com/openai/v1/chat/completions"
