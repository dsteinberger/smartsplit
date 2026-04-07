"""Mistral provider — free tier, best for translation."""

from __future__ import annotations

from smartsplit.providers.base import OpenAICompatibleProvider


class MistralProvider(OpenAICompatibleProvider):
    name = "mistral"
    api_url = "https://api.mistral.ai/v1/chat/completions"
