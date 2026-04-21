"""Anthropic Claude provider — paid, best for code and reasoning."""

from __future__ import annotations

import httpx

from smartsplit.exceptions import ProviderError
from smartsplit.models import TokenUsage
from smartsplit.providers.anthropic_adapter import (
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    anthropic_to_openai,
    openai_to_anthropic,
)
from smartsplit.providers.base import _EMPTY_USAGE, LLMProvider, http_error_to_provider_error

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    native_format = "anthropic"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        messages: list[dict[str, str]] | None = None,
        extra_body: dict | None = None,
    ) -> tuple[str, TokenUsage]:
        # Anthropic format: separate system messages from user/assistant messages
        api_messages = []
        system_text = ""
        for msg in messages or [{"role": "user", "content": prompt}]:
            if msg["role"] == "system":
                system_text += msg["content"] + "\n"
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
        if not api_messages:
            api_messages = [{"role": "user", "content": prompt}]

        body: dict = {
            "model": model or self.config.model,
            "max_tokens": self.config.max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS,
            "messages": api_messages,
        }
        if system_text.strip():
            body["system"] = system_text.strip()

        try:
            response = await self.http.post(_ANTHROPIC_URL, headers=self._headers(), json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise http_error_to_provider_error(self.name, e) from e
        except httpx.TimeoutException as e:
            raise ProviderError(self.name, "Request timed out") from e
        try:
            data = response.json()
        except ValueError as e:
            raise ProviderError(self.name, "Invalid JSON in API response") from e
        content = self._extract(data, "content", 0, "text")
        usage_data = data.get("usage")
        if usage_data:
            prompt_t = usage_data.get("input_tokens", 0)
            completion_t = usage_data.get("output_tokens", 0)
            usage = TokenUsage(
                prompt_tokens=prompt_t, completion_tokens=completion_t, total_tokens=prompt_t + completion_t
            )
        else:
            usage = _EMPTY_USAGE
        return content, usage

    async def proxy_openai_request(self, body: dict) -> dict:
        """Translate OpenAI body → Anthropic Messages API, forward, translate response back."""
        model = self.config.model
        anthropic_body = openai_to_anthropic({k: v for k, v in body.items() if k != "stream"}, model)
        try:
            response = await self.http.post(_ANTHROPIC_URL, headers=self._headers(), json=anthropic_body)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise http_error_to_provider_error(self.name, e) from e
        except httpx.TimeoutException as e:
            raise ProviderError(self.name, "Request timed out") from e
        try:
            data = response.json()
        except ValueError as e:
            raise ProviderError(self.name, "Invalid JSON in API response") from e
        return anthropic_to_openai(data, model)
