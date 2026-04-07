"""HuggingFace Inference provider — free backup via OpenAI-compatible endpoint."""

from __future__ import annotations

from smartsplit.providers.base import OpenAICompatibleProvider


class HuggingFaceProvider(OpenAICompatibleProvider):
    name = "huggingface"
    api_url = "https://router.huggingface.co/v1/chat/completions"
