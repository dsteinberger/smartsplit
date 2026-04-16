"""Provider registry — discovers, instantiates, and manages providers."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING

import httpx

from smartsplit.exceptions import NoProviderAvailableError, ProviderError
from smartsplit.models import ContextTier
from smartsplit.providers.base import BaseProvider, LLMProvider, SearchProvider

if TYPE_CHECKING:
    from smartsplit.config import ProviderConfig
    from smartsplit.models import TokenUsage

# Max prompt chars per context tier — controls how much context workers receive.
# Smaller tiers save tokens for free-tier providers with tight rate limits.
_CONTEXT_TIER_MAX_CHARS: dict[ContextTier, int] = {
    ContextTier.SMALL: 4_000,  # ~1k tokens — Cerebras, Groq free
    ContextTier.MEDIUM: 16_000,  # ~4k tokens — Mistral, OpenRouter
    ContextTier.LARGE: 64_000,  # ~16k tokens — Gemini, paid APIs
}

logger = logging.getLogger("smartsplit.registry")

_KEY_PATTERN = re.compile(r"(sk-ant-|gsk_|AIza|tvly_|srp_|sk-|csk-|dsk-|mis-)\S+|\b[A-Za-z0-9_-]{32,}\b")


# ── Circuit breaker ────────────────────────────────────────────

_PROVIDER_CALL_TIMEOUT = 30  # seconds — max wait per provider call
_CB_FAILURE_THRESHOLD = 5  # weighted failure points before marking unhealthy
_CB_FAILURE_WINDOW = 120  # 2 minutes window
_CB_BASE_TIMEOUT = 60  # base cooldown (seconds)
_CB_MAX_TIMEOUT = 1800  # max cooldown: 30 minutes
_CB_SUCCESSES_TO_CLOSE = 3  # consecutive successes to fully reset backoff


def _failure_weight(exc: Exception | None) -> float:
    """Return failure weight based on error type.

    - Auth errors (401/403): immediate trip (weight = threshold)
    - Rate limits (429): half weight (need ~10 to trip)
    - Timeouts / server errors / other: normal weight (1.0)
    """
    if exc is None:
        return 1.0
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            return float(_CB_FAILURE_THRESHOLD)
        if status == 429:
            return 0.5
    return 1.0


class CircuitBreaker:
    """Per-provider circuit breaker with exponential backoff and half-open probing.

    States:
    - CLOSED: provider is healthy, all requests go through.
    - OPEN: provider tripped after ``_CB_FAILURE_THRESHOLD`` weighted failure
      points within ``_CB_FAILURE_WINDOW``. Cooldown grows exponentially
      (60s → 120s → 240s … up to 30 min).
    - HALF-OPEN: cooldown expired, exactly one probe request is allowed.
      If the probe succeeds → CLOSED. If it fails → OPEN (immediate re-trip).

    Backoff resets gradually: ``_CB_SUCCESSES_TO_CLOSE`` consecutive successes
    are needed to clear the trip counter (not a single success).
    """

    def __init__(self) -> None:
        self._failures: dict[str, list[tuple[float, float]]] = {}  # (timestamp, weight)
        self._open_until: dict[str, float] = {}
        self._consecutive_trips: dict[str, int] = {}
        self._half_open: set[str] = set()
        self._consecutive_successes: dict[str, int] = {}

    def record_failure(self, provider: str, exc: Exception | None = None) -> None:
        """Record a provider failure and trip the breaker if threshold is reached."""
        now = time.time()
        weight = _failure_weight(exc)
        self._consecutive_successes.pop(provider, None)

        # Half-open probe failed → reopen immediately
        if provider in self._half_open:
            self._half_open.discard(provider)
            trips = self._consecutive_trips.get(provider, 0) + 1
            self._consecutive_trips[provider] = trips
            timeout = min(_CB_BASE_TIMEOUT * (2 ** (trips - 1)), _CB_MAX_TIMEOUT)
            self._open_until[provider] = now + timeout
            self._failures[provider] = []
            logger.warning("Half-open probe FAILED for %s — reopening for %ss (trip #%s)", provider, timeout, trips)
            return

        if provider not in self._failures:
            self._failures[provider] = []
        self._failures[provider].append((now, weight))
        # Trim old failures outside the window
        cutoff = now - _CB_FAILURE_WINDOW
        self._failures[provider] = [(t, w) for t, w in self._failures[provider] if t > cutoff]
        # Check threshold (sum of weights)
        total = sum(w for _, w in self._failures[provider])
        if total >= _CB_FAILURE_THRESHOLD:
            trips = self._consecutive_trips.get(provider, 0) + 1
            self._consecutive_trips[provider] = trips
            timeout = min(_CB_BASE_TIMEOUT * (2 ** (trips - 1)), _CB_MAX_TIMEOUT)
            self._open_until[provider] = now + timeout
            self._failures[provider] = []
            logger.warning("Circuit breaker OPEN for %s — skipping for %ss (trip #%s)", provider, timeout, trips)

    def record_success(self, provider: str) -> None:
        """Record a successful call. Closes a half-open breaker or counts toward reset."""
        if provider in self._half_open:
            self._half_open.discard(provider)
            logger.info("Half-open probe OK for %s — circuit breaker CLOSED", provider)

        self._failures.pop(provider, None)

        # Graduated reset: need N consecutive successes to clear the trip counter
        count = self._consecutive_successes.get(provider, 0) + 1
        self._consecutive_successes[provider] = count
        if count >= _CB_SUCCESSES_TO_CLOSE:
            self._consecutive_trips.pop(provider, None)
            self._consecutive_successes.pop(provider, None)

    def is_healthy(self, provider: str) -> bool:
        """Return True if the provider accepts requests (CLOSED or entering HALF-OPEN)."""
        # A probe is already in flight — block additional requests
        if provider in self._half_open:
            return False
        deadline = self._open_until.get(provider)
        if deadline is None:
            return True
        if time.time() > deadline:
            self._open_until.pop(provider, None)
            self._half_open.add(provider)
            logger.info("Circuit breaker HALF-OPEN for %s — allowing probe request", provider)
            return True
        return False

    def get_unhealthy(self) -> list[str]:
        """Return names of providers whose circuit breaker is currently open or half-open."""
        now = time.time()
        open_providers = [p for p, t in self._open_until.items() if now <= t]
        return open_providers + [p for p in self._half_open if p not in open_providers]


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
    "perplexity": ("smartsplit.providers.perplexity", "PerplexityProvider"),
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
        brain_name: str = "",
    ) -> None:
        from smartsplit.config import DEFAULT_FREE_LLM_PRIORITY

        self._providers: dict[str, BaseProvider] = {}
        self._provider_configs = provider_configs
        self._http = http
        self._free_llm_priority = free_llm_priority or list(DEFAULT_FREE_LLM_PRIORITY)
        self.brain_name = brain_name
        self.circuit_breaker = CircuitBreaker()

        for name, pconfig in provider_configs.items():
            if not pconfig.enabled or not pconfig.api_key:
                continue
            cls = self._resolve_class(name)
            if cls is None:
                logger.warning("Unknown provider %r, skipping", name)
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
            logger.info("Initialized providers: %s", enabled_names)
            if not llm_names:
                logger.warning(
                    "No LLM provider active. Requests will fail. Set GROQ_API_KEY (free: groq.com) or GEMINI_API_KEY."
                )
            if not search_names:
                logger.info(
                    "No search provider active. web_search will not work. "
                    "Set SERPER_API_KEY (free: serper.dev) for web search."
                )
            if self.brain_name and self.brain_name in self._providers:
                worker_names = [n for n in llm_names if n != self.brain_name]
                logger.info("Brain: %s | Workers: %s", self.brain_name, worker_names)
            elif self.brain_name:
                logger.warning("Brain %r not available, will use fallback routing", self.brain_name)

    # ── Lookups ──────────────────────────────────────────────

    def get_all(self) -> dict[str, BaseProvider]:
        """Return a copy of all registered providers."""
        return dict(self._providers)

    def get_llm_providers(self) -> dict[str, LLMProvider]:
        """Return all registered LLM providers."""
        return {n: p for n, p in self._providers.items() if isinstance(p, LLMProvider)}

    def get_search_providers(self) -> dict[str, SearchProvider]:
        """Return all registered search providers."""
        return {n: p for n, p in self._providers.items() if isinstance(p, SearchProvider)}

    def get(self, name: str) -> BaseProvider | None:
        """Look up a provider by name."""
        return self._providers.get(name)

    # ── High-level helpers ───────────────────────────────────

    async def call_brain(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
    ) -> tuple[str, TokenUsage]:
        """Call the brain (main LLM) with fallback.

        The brain is the primary LLM that produces the final response.
        If the brain is down, falls back to the next best provider.
        """
        from smartsplit.config import _BRAIN_PRIORITY

        order = [self.brain_name] if self.brain_name else []
        order += [p for p in _BRAIN_PRIORITY if p != self.brain_name and p in self._providers]

        for name in order:
            provider = self._providers.get(name)
            if provider is None or not isinstance(provider, LLMProvider):
                continue
            if not self.circuit_breaker.is_healthy(name):
                logger.info("Skipping brain candidate %s — circuit breaker open", name)
                continue
            try:
                async with asyncio.timeout(_PROVIDER_CALL_TIMEOUT):
                    result, usage = await provider.complete(prompt, messages=messages)
                self.circuit_breaker.record_success(name)
                if name == self.brain_name:
                    logger.debug("Brain %r responded successfully", name)
                else:
                    logger.warning("Brain %r unavailable, used fallback: %s", self.brain_name, name)
                return result, usage
            except TimeoutError as e:
                logger.warning("Brain candidate %r timed out after %ss", name, _PROVIDER_CALL_TIMEOUT)
                self.circuit_breaker.record_failure(name, e)
            except Exception as e:
                logger.warning("Brain candidate %r failed: %s: %s", name, type(e).__name__, _sanitize_error(e))
                self.circuit_breaker.record_failure(name, e)

        raise NoProviderAvailableError("brain")

    async def proxy_to_brain(self, body: dict) -> dict:
        """Forward a raw OpenAI-compatible request body to the brain and return the raw response.

        Used when the request contains tools/tool_choice that must be preserved
        for the agent loop. The body's messages may have been enriched by SmartSplit,
        but everything else (tools, stream, etc.) is passed through unchanged.
        """
        from smartsplit.config import _BRAIN_PRIORITY

        order = [self.brain_name] if self.brain_name else []
        order += [p for p in _BRAIN_PRIORITY if p != self.brain_name and p in self._providers]

        for name in order:
            provider = self._providers.get(name)
            if provider is None or not isinstance(provider, LLMProvider):
                continue
            if not self.circuit_breaker.is_healthy(name):
                continue
            try:
                async with asyncio.timeout(_PROVIDER_CALL_TIMEOUT):
                    data = await provider.proxy_openai_request(body)
                self.circuit_breaker.record_success(name)
                logger.info("Proxied to brain %r successfully", name)
                return data
            except NotImplementedError:
                logger.debug("Provider %r does not support agent-mode passthrough, skipping", name)
                continue
            except TimeoutError as e:
                logger.warning("Brain proxy %r timed out", name)
                self.circuit_breaker.record_failure(name, e)
            except Exception as e:
                logger.warning("Brain proxy %r failed: %s: %s", name, type(e).__name__, _sanitize_error(e))
                self.circuit_breaker.record_failure(name, e)

        raise NoProviderAvailableError("brain")

    async def call_free_llm(self, prompt: str, prefer: str = "groq") -> str:
        """Try free LLMs in priority order, with brain as last-resort fallback."""
        order = [prefer] + [p for p in self._free_llm_priority if p != prefer]
        # Brain is excluded from the main loop but added as last-resort fallback
        # so single-provider setups (only a paid brain) still work.
        if self.brain_name and self.brain_name not in order:
            order.append(self.brain_name)

        for name in order:
            provider = self._providers.get(name)
            if provider is None or not isinstance(provider, LLMProvider):
                continue
            if not self.circuit_breaker.is_healthy(name):
                logger.info("Skipping %s — circuit breaker open", name)
                continue
            # Truncate prompt to provider's context tier
            pconfig = self._provider_configs.get(name)
            tier = pconfig.context_tier if pconfig else ContextTier.SMALL
            max_chars = _CONTEXT_TIER_MAX_CHARS[tier]
            truncated = prompt[:max_chars] if len(prompt) > max_chars else prompt
            if len(prompt) > max_chars:
                logger.debug("Truncated prompt for %s: %d → %d chars (tier %s)", name, len(prompt), max_chars, tier)
            try:
                async with asyncio.timeout(_PROVIDER_CALL_TIMEOUT):
                    result, _usage = await provider.complete(truncated)
                self.circuit_breaker.record_success(name)
                return result
            except TimeoutError as e:
                logger.warning("Free LLM %s timed out after %ss", name, _PROVIDER_CALL_TIMEOUT)
                self.circuit_breaker.record_failure(name, e)
            except Exception as e:
                logger.warning("Free LLM %s failed: %s: %s", name, type(e).__name__, _sanitize_error(e))
                self.circuit_breaker.record_failure(name, e)

        raise NoProviderAvailableError("free_llm")

    async def call_llm(
        self,
        name: str,
        prompt: str,
        model: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> tuple[str, TokenUsage]:
        """Call a specific LLM provider by name."""
        provider = self._providers.get(name)
        if provider is None or not isinstance(provider, LLMProvider):
            raise ProviderError(name, "not available or not an LLM provider")
        return await provider.complete(prompt, model=model, messages=messages)

    async def call_search(self, name: str, query: str) -> str:
        """Call a specific search provider by name."""
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
