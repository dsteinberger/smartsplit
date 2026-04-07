"""Anthropic Claude provider — paid, best for code and reasoning."""

from __future__ import annotations

import httpx

from smartsplit.exceptions import ProviderError
from smartsplit.models import TokenUsage
from smartsplit.providers.base import _EMPTY_USAGE, LLMProvider, _http_error_message


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        messages: list[dict[str, str]] | None = None,
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
            "max_tokens": self.config.max_tokens,
            "messages": api_messages,
        }
        if system_text.strip():
            body["system"] = system_text.strip()

        try:
            response = await self.http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ProviderError(self.name, _http_error_message(e)) from e
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
