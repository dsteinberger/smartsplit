"""Google Gemini provider — free tier, strong for summaries, reasoning, and code."""

from __future__ import annotations

import httpx

from smartsplit.exceptions import ProviderError
from smartsplit.models import TokenUsage
from smartsplit.providers.base import _EMPTY_USAGE, LLMProvider, _http_error_message

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_GEMINI_OPENAI_COMPAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


class GeminiProvider(LLMProvider):
    """Google Gemini provider — free tier, strong for summaries and reasoning."""

    name = "gemini"

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        messages: list[dict[str, str]] | None = None,
        extra_body: dict | None = None,
    ) -> tuple[str, TokenUsage]:
        url = _GEMINI_URL.format(model=model or self.config.model)

        # Convert messages to Gemini format, merging consecutive same-role messages
        if messages:
            contents: list[dict[str, str | list[dict[str, str]]]] = []
            for msg in messages:
                role = "model" if msg["role"] == "assistant" else "user"
                text = msg["content"]
                # Gemini forbids consecutive messages with same role — merge them
                if contents and contents[-1]["role"] == role:
                    contents[-1]["parts"][0]["text"] += "\n\n" + text
                else:
                    contents.append({"role": role, "parts": [{"text": text}]})
        else:
            contents = [{"parts": [{"text": prompt}]}]

        try:
            response = await self.http.post(
                url,
                headers={"x-goog-api-key": self.api_key},
                json={
                    "contents": contents,
                    "generationConfig": {
                        "temperature": self.config.temperature,
                        "maxOutputTokens": self.config.max_tokens,
                    },
                    # Disable tool use — Gemini may attempt grounding/search tool calls
                    # which return no text content. SmartSplit handles search externally.
                    "toolConfig": {"functionCallingConfig": {"mode": "NONE"}},
                },
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

        # Gemini may return candidates without content (safety filter blocks)
        candidates = data.get("candidates", [])
        if not candidates:
            reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
            raise ProviderError(self.name, f"No candidates returned (blockReason={reason})")

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "")

        # Gemini omits 'content' when blocked (SAFETY, RECITATION, OTHER, etc.)
        if "content" not in candidate:
            raise ProviderError(self.name, f"No content in response (finishReason={finish_reason})")

        content = self._extract(data, "candidates", 0, "content", "parts", 0, "text")

        usage_meta = data.get("usageMetadata")
        if usage_meta:
            prompt_t = usage_meta.get("promptTokenCount", 0)
            completion_t = usage_meta.get("candidatesTokenCount", 0)
            usage = TokenUsage(
                prompt_tokens=prompt_t, completion_tokens=completion_t, total_tokens=prompt_t + completion_t
            )
        else:
            usage = _EMPTY_USAGE

        return content, usage

    async def proxy_openai_request(self, body: dict) -> dict:
        """Forward via Gemini's OpenAI-compatible endpoint (needed for tool use passthrough)."""
        proxy_body = dict(body)
        proxy_body["model"] = self.config.model
        proxy_body.pop("stream", None)
        try:
            response = await self.http.post(
                _GEMINI_OPENAI_COMPAT_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=proxy_body,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ProviderError(self.name, _http_error_message(e)) from e
        except httpx.TimeoutException as e:
            raise ProviderError(self.name, "Request timed out") from e
        try:
            return response.json()
        except ValueError as e:
            raise ProviderError(self.name, "Invalid JSON in API response") from e
