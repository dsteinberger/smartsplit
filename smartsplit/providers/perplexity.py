"""Perplexity provider — paid, search-augmented LLM (Sonar models)."""

from __future__ import annotations

from smartsplit.providers.base import OpenAICompatibleProvider


class PerplexityProvider(OpenAICompatibleProvider):
    name = "perplexity"
    api_url = "https://api.perplexity.ai/chat/completions"
