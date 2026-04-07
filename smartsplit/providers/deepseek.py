"""DeepSeek provider — very cheap, strong for code and reasoning."""

from __future__ import annotations

from smartsplit.providers.base import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    name = "deepseek"
    api_url = "https://api.deepseek.com/chat/completions"
