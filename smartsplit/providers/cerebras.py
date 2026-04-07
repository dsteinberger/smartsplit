"""Cerebras provider — 6x faster than Groq, free 1M tokens/day, Qwen3-235B."""

from __future__ import annotations

from smartsplit.providers.base import OpenAICompatibleProvider


class CerebrasProvider(OpenAICompatibleProvider):
    name = "cerebras"
    api_url = "https://api.cerebras.ai/v1/chat/completions"
    model = "qwen-3-235b-a22b-instruct-2507"
