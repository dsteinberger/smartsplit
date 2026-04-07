"""Provider registry — discovers, instantiates, and manages providers."""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

import httpx

from smartsplit.exceptions import NoProviderAvailableError, ProviderError
from smartsplit.providers.base import BaseProvider, LLMProvider, SearchProvider

if TYPE_CHECKING:
    from smartsplit.models import TokenUsage

if TYPE_CHECKING:
    from smartsplit.config import ProviderConfig

logger = logging.getLogger("smartsplit.registry")

_KEY_PATTERN = re.compile(r"(sk-ant-|gsk_|AIza|tvly_|srp_|sk-|csk-|dsk-|mis-)\S+|\b[A-Za-z0-9_-]{32,}\b")

# ── Circuit breaker ────────────────────────────────────────────

_CB_FAILURE_THRESHOLD = 3  # failures before marking unhealthy
_CB_FAILURE_WINDOW = 300  # 5 minutes window
_CB_RECOVERY_TIMEOUT = 1800  # 30 minutes cooldown


class CircuitBreaker:
    """Per-provider circuit breaker. Marks providers unhealthy after repeated failures."""

    def __init__(self) -> None:
        self._failures: dict[str, list[float]] = {}
        self._open_until: dict[str, float] = {}

    def record_failure(self, provider: str) -> None:
        now = time.time()
        if provider not in self._failures:
            self._failures[provider] = []
        self._failures[provider].append(now)
        # Trim old failures outside the window
        cutoff = now - _CB_FAILURE_WINDOW
        self._failures[provider] = [t for t in self._failures[provider] if t > cutoff]
        # Check threshold
        if len(self._failures[provider]) >= _CB_FAILURE_THRESHOLD:
            self._open_until[provider] = now + _CB_RECOVERY_TIMEOUT
            self._failures[provider] = []
            logger.warning(f"Circuit breaker OPEN for {provider} — skipping for {_CB_RECOVERY_TIMEOUT}s")

    def record_success(self, provider: str) -> None:
        self._failures.pop(provider, None)

    def is_healthy(self, provider: str) -> bool:
        deadline = self._open_until.get(provider)
        if deadline is None:
            return True
        if time.time() > deadline:
            self._open_until.pop(provider, None)
            logger.info(f"Circuit breaker CLOSED for {provider} — recovered")
            return True
        return False

    def get_unhealthy(self) -> list[str]:
        now = time.time()
        return [p for p, t in self._open_until.items() if now <= t]


def _sanitize_error(exc: Exception) -> str:
    """Remove potential API keys from error messages."""
    return _KEY_PATTERN.sub("[REDACTED]", str(exc))


# Lazy import map — avoids circular imports and keeps this module lightweight.
_PROVIDER_CLASSES: dict[str, tuple[str, str]] = {
    "groq": ("smartsplit.providers.groq", "GroqProvider"),
    "cerebras": ("smartsplit.providers.cerebras", "CerebrasProvider"),
    "gemini": ("smartsplit.providers.gemini", "GeminiProvider"),
    "mistral": ("smartsplit.providers.mistral", "MistralProvider"),
    "deepseek": ("smartsplit.providers.deepseek", "DeepSeekProvider"),
    "openrouter": ("smartsplit.providers.openrouter", "OpenRouterProvider"),
    "anthropic": ("smartsplit.providers.anthropic", "AnthropicProvider"),
    "openai": ("smartsplit.providers.openai", "OpenAIProvider"),
    "huggingface": ("smartsplit.providers.huggingface", "HuggingFaceProvider"),
    "cloudflare": ("smartsplit.providers.cloudflare", "CloudflareProvider"),
    "serper": ("smartsplit.providers.serper", "SerperProvider"),
    "tavily": ("smartsplit.providers.tavily", "TavilyProvider"),
}


class ProviderRegistry:
    """Central registry that holds live provider instances.

    Responsibilities:
    - Instantiate providers from config (Factory)
    - Provide typed lookups (LLM vs Search)
    - Offer a resilient ``call_free_llm`` with ordered fallback
    """

    def __init__(
        self,
        provider_configs: dict[str, ProviderConfig],
        http: httpx.AsyncClient,
        free_llm_priority: list[str] | None = None,
    ) -> None:
        from smartsplit.config import DEFAULT_FREE_LLM_PRIORITY

        self._providers: dict[str, BaseProvider] = {}
        self._free_llm_priority = free_llm_priority or list(DEFAULT_FREE_LLM_PRIORITY)
        self.circuit_breaker = CircuitBreaker()

        for name, pconfig in provider_configs.items():
            if not pconfig.enabled or not pconfig.api_key:
                continue
            cls = self._resolve_class(name)
            if cls is None:
                logger.warning(f"Unknown provider '{name}', skipping")
                continue
            self._providers[name] = cls(config=pconfig, http=http)

        # ── Startup diagnostics ────────────────────────────────
        enabled_names = list(self._providers)
        if not enabled_names:
            logger.warning(
                "No providers configured! SmartSplit cannot route any requests. "
                "Set at least GROQ_API_KEY (free: groq.com) to get started."
            )
        else:
            llm_names = [n for n, p in self._providers.items() if isinstance(p, LLMProvider)]
            search_names = [n for n, p in self._providers.items() if isinstance(p, SearchProvider)]
            logger.info(f"Initialized providers: {enabled_names}")
            if not llm_names:
                logger.warning(
                    "No LLM provider active. Requests will fail. Set GROQ_API_KEY (free: groq.com) or GEMINI_API_KEY."
                )
            if not search_names:
                logger.info(
                    "No search provider active. web_search will not work. "
                    "Set SERPER_API_KEY (free: serper.dev) for web search."
                )

    # ── Lookups ──────────────────────────────────────────────

    def get_all(self) -> dict[str, BaseProvider]:
        return dict(self._providers)

    def get_llm_providers(self) -> dict[str, LLMProvider]:
        return {n: p for n, p in self._providers.items() if isinstance(p, LLMProvider)}

    def get_search_providers(self) -> dict[str, SearchProvider]:
        return {n: p for n, p in self._providers.items() if isinstance(p, SearchProvider)}

    def get(self, name: str) -> BaseProvider | None:
        return self._providers.get(name)

    # ── High-level helpers ───────────────────────────────────

    async def call_free_llm(self, prompt: str, prefer: str = "groq") -> str:
        """Try free LLMs in priority order, with fallback and circuit breaker."""
        order = [prefer] + [p for p in self._free_llm_priority if p != prefer]

        for name in order:
            provider = self._providers.get(name)
            if provider is None or not isinstance(provider, LLMProvider):
                continue
            if not self.circuit_breaker.is_healthy(name):
                logger.info(f"Skipping {name} — circuit breaker open")
                continue
            try:
                result, _usage = await provider.complete(prompt)
                self.circuit_breaker.record_success(name)
                return result
            except Exception as e:
                logger.warning(f"Free LLM {name} failed: {type(e).__name__}: {_sanitize_error(e)}")
                self.circuit_breaker.record_failure(name)

        raise NoProviderAvailableError("free_llm")

    async def call_llm(
        self,
        name: str,
        prompt: str,
        model: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> tuple[str, TokenUsage]:
        provider = self._providers.get(name)
        if provider is None or not isinstance(provider, LLMProvider):
            raise ProviderError(name, "not available or not an LLM provider")
        return await provider.complete(prompt, model=model, messages=messages)

    async def call_search(self, name: str, query: str) -> str:
        provider = self._providers.get(name)
        if provider is None or not isinstance(provider, SearchProvider):
            raise ProviderError(name, "not available or not a search provider")
        return await provider.search(query)

    # ── Internal ─────────────────────────────────────────────

    @staticmethod
    def _resolve_class(name: str) -> type[BaseProvider] | None:
        entry = _PROVIDER_CLASSES.get(name)
        if entry is None:
            return None
        module_path, class_name = entry
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, class_name)
