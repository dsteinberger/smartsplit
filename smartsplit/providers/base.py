"""Abstract base classes for providers (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import httpx

from smartsplit.exceptions import ProviderError
from smartsplit.models import TokenUsage

if TYPE_CHECKING:
    from smartsplit.config import ProviderConfig


_EMPTY_USAGE = TokenUsage()


def _http_error_message(e: httpx.HTTPStatusError) -> str:
    """Build a ProviderError message from an HTTP error, including API detail for context length detection."""
    try:
        detail = e.response.json().get("error", {}).get("message", "")[:200]
    except Exception:
        detail = ""
    msg = f"HTTP {e.response.status_code}"
    if detail:
        msg += f": {detail}"
    return msg


def _extract_usage(data: dict) -> TokenUsage:
    """Extract token usage from an OpenAI-compatible API response."""
    usage = data.get("usage")
    if not usage:
        return _EMPTY_USAGE
    return TokenUsage(
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )


class BaseProvider(ABC):
    """Base class for all providers."""

    name: str

    def __init__(self, config: ProviderConfig, http: httpx.AsyncClient) -> None:
        self.config = config
        self.api_key = config.api_key
        self.http = http

    def _extract(self, data: object, *keys: str | int) -> str:
        """Safely extract a nested value from a parsed API response.

        Raises ProviderError with a clear message if the path doesn't exist.
        """
        current = data
        path_so_far: list[str] = []
        for key in keys:
            path_so_far.append(str(key))
            try:
                current = current[key]
            except (KeyError, IndexError, TypeError) as exc:
                raise ProviderError(
                    self.name,
                    f"Unexpected API response structure at {'.'.join(path_so_far)}",
                ) from exc
        if not isinstance(current, str):
            raise ProviderError(
                self.name,
                f"Expected string at {'.'.join(path_so_far)}, got {type(current).__name__}",
            )
        return current


class LLMProvider(BaseProvider):
    """A provider that generates text from a prompt."""

    # Native wire format for this provider's API. Used to decide whether the brain
    # can accept an Anthropic-format request as-is or needs translation.
    native_format: str = "openai"

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        messages: list[dict[str, str]] | None = None,
        extra_body: dict | None = None,
    ) -> tuple[str, TokenUsage]:
        """Send a prompt and return (completion_text, token_usage).

        If *messages* is provided, send the full conversation history.
        Otherwise, wrap *prompt* in a single user message.
        If *model* is provided, use it instead of the provider's default.
        If *extra_body* is provided, merge it into the API request body
        (used to pass through tools, tool_choice, etc. from the client).
        """

    async def proxy_openai_request(self, body: dict) -> dict:
        """Forward an OpenAI-format chat completions body and return an OpenAI-format response.

        Providers with a non-OpenAI native protocol (Anthropic) translate in/out via an
        adapter. Providers with OpenAI-compatible endpoints override to point at the
        right URL. Used by the agent-mode proxy to preserve tools/tool_choice.

        Default: raise — concrete providers must opt in by overriding.
        """
        raise NotImplementedError(f"Provider {self.name!r} does not support proxy_openai_request")


class OpenAICompatibleProvider(LLMProvider):
    """Base for providers using the OpenAI chat completions API format.

    Subclasses only need to set ``name`` and ``api_url``.
    Model, temperature, and max_tokens come from ``self.config``.
    """

    api_url: str

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        messages: list[dict[str, str]] | None = None,
        extra_body: dict | None = None,
    ) -> tuple[str, TokenUsage]:
        body: dict = {
            "model": model or self.config.model,
            "messages": messages or [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if extra_body:
            body.update(extra_body)
        try:
            response = await self.http.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
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
        content = self._extract(data, "choices", 0, "message", "content")
        return content, _extract_usage(data)

    async def proxy_openai_request(self, body: dict) -> dict:
        """Forward an OpenAI-format body to ``self.api_url`` unchanged (beyond model override)."""
        proxy_body = dict(body)
        proxy_body["model"] = self.config.model
        proxy_body.pop("stream", None)
        try:
            response = await self.http.post(
                self.api_url,
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


class SearchProvider(BaseProvider):
    """A provider that returns web search results."""

    @abstractmethod
    async def search(self, query: str) -> str:
        """Execute a search query and return formatted results."""
