"""Router — scores providers and routes each subtask to the best one.

Scoring formula (additive weighted):
  Score = w_quality × Quality + w_cost × CostScore + w_availability × AvailabilityScore

Weights vary by mode (economy favors cost, quality favors quality).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from smartsplit.models import (
    Complexity,
    EscalationRecord,
    Mode,
    ProviderType,
    RouteResult,
    Subtask,
    TaskType,
    TerminationState,
)
from smartsplit.providers.base import SearchProvider
from smartsplit.quota import estimate_tokens

if TYPE_CHECKING:
    from smartsplit.config import SmartSplitConfig
    from smartsplit.providers.registry import ProviderRegistry
    from smartsplit.quota import QuotaTracker

logger = logging.getLogger("smartsplit.router")

# Weights per mode: (w_quality, w_cost, w_availability)
_MODE_WEIGHTS: dict[Mode, tuple[float, float, float]] = {
    Mode.ECONOMY: (0.3, 0.5, 0.2),  # cost dominates
    Mode.BALANCED: (0.5, 0.3, 0.2),  # quality dominates slightly
    Mode.QUALITY: (0.7, 0.1, 0.2),  # quality dominates heavily
}

# Complexity adjustments to quality weight
_COMPLEXITY_QUALITY_BOOST: dict[Complexity, float] = {
    Complexity.HIGH: 0.15,  # high complexity → quality matters more
    Complexity.MEDIUM: 0.0,
    Complexity.LOW: -0.10,  # low complexity → quality matters less, cost more
}

# Availability threshold: below this ratio, provider is considered "at risk"
_AVAILABILITY_DANGER_THRESHOLD = 0.10

# Quality gate: minimum response length by complexity (chars).
# Responses shorter than this are considered low-quality and trigger escalation.
_MIN_RESPONSE_LENGTH: dict[Complexity, int] = {
    Complexity.LOW: 5,
    Complexity.MEDIUM: 20,
    Complexity.HIGH: 50,
}

# Quality gate error patterns — checked against the first 200 chars of the response.
# These indicate refusal or system errors, not legitimate content.
_ERROR_PATTERNS = [
    # Refusals
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "i'm not able",
    "i am not able",
    "i don't have access",
    "i do not have access",
    "i'm sorry, but i",
    "i apologize, but i",
    "as an ai language model",
    "as an ai,",
    "as a language model",
    "i'm not able to help",
    # Rate limits / errors
    "rate limit exceeded",
    "quota exceeded",
    "error: rate limit",
    "error: 429",
    "error: 503",
    "error: 500",
    "internal server error",
    "service unavailable",
]

# How many leading characters to scan for error patterns.
_ERROR_SCAN_LENGTH = 200

# Quality gates are skipped in economy mode for speed.
_QUALITY_GATE_MODES = {Mode.BALANCED, Mode.QUALITY}


def _no_provider_hint(task_type: TaskType) -> str:
    """User-friendly hint when no provider can handle a task type."""
    hints = {
        TaskType.WEB_SEARCH: "Set SERPER_API_KEY (free: serper.dev) or TAVILY_API_KEY for web search.",
        TaskType.CODE: "Set GROQ_API_KEY (free: groq.com) or ANTHROPIC_API_KEY for code generation.",
        TaskType.REASONING: "Set GEMINI_API_KEY (free: ai.google.dev) or ANTHROPIC_API_KEY for reasoning.",
        TaskType.TRANSLATION: "Set MISTRAL_API_KEY (free: console.mistral.ai) for translation.",
        TaskType.SUMMARIZE: "Set GROQ_API_KEY (free: groq.com) or GEMINI_API_KEY for summarization.",
        TaskType.MATH: "Set DEEPSEEK_API_KEY or ANTHROPIC_API_KEY for math tasks.",
        TaskType.CREATIVE: "Set ANTHROPIC_API_KEY for creative writing tasks.",
        TaskType.FACTUAL: "Set GROQ_API_KEY (free: groq.com) or GEMINI_API_KEY for factual queries.",
        TaskType.EXTRACTION: "Set GROQ_API_KEY (free: groq.com) or GEMINI_API_KEY for data extraction.",
    }
    return hints.get(task_type, "Set GROQ_API_KEY (free: groq.com) to get started.")


class Router:
    """Routes ``Subtask`` objects to the best available provider."""

    def __init__(
        self,
        registry: ProviderRegistry,
        quota: QuotaTracker,
        config: SmartSplitConfig,
    ) -> None:
        self._registry = registry
        self._quota = quota
        self._config = config

    def score(
        self,
        provider_name: str,
        provider_type: ProviderType,
        subtask: Subtask,
        mode: Mode,
    ) -> float:
        """Compute the routing score for one (provider, subtask) pair.

        Uses an additive weighted formula so that one weak factor
        doesn't zero out the entire score (unlike the old multiplicative approach).
        """
        # Base weights from mode
        w_quality, w_cost, w_avail = _MODE_WEIGHTS.get(mode, _MODE_WEIGHTS[Mode.BALANCED])

        # Complexity shifts weight between quality and cost
        boost = _COMPLEXITY_QUALITY_BOOST.get(subtask.complexity, 0.0)
        w_quality = min(1.0, max(0.0, w_quality + boost))
        w_cost = min(1.0, max(0.0, w_cost - boost))

        # Quality: competence table score [0.0 - 1.0]
        competence = self._config.competence_table.get(
            subtask.type.value,
            self._config.competence_table.get("general", {}),
        )
        quality = competence.get(provider_name, 5) / 10.0

        # Cost: free = 1.0 (best), paid = 0.0 (worst)
        cost_score = 1.0 if provider_type == ProviderType.FREE else 0.0

        # Availability: 1.0 if above danger threshold, degrades below it
        raw_avail = self._quota.get_availability(provider_name)
        if raw_avail >= _AVAILABILITY_DANGER_THRESHOLD:
            avail_score = 1.0
        elif raw_avail <= 0.0:
            avail_score = 0.0
        else:
            avail_score = raw_avail / _AVAILABILITY_DANGER_THRESHOLD

        return w_quality * quality + w_cost * cost_score + w_avail * avail_score

    @staticmethod
    def passes_quality_gate(response: str, subtask: Subtask) -> bool:
        """Check if a response meets minimum quality standards.

        Returns True if the response is acceptable, False if it should be
        escalated to the next provider.
        """
        if not response or not response.strip():
            return False

        # Length check based on complexity
        min_len = _MIN_RESPONSE_LENGTH.get(subtask.complexity, 5)
        if len(response.strip()) < min_len:
            return False

        # Error pattern check — only scan the beginning to avoid false positives
        response_start = response[:_ERROR_SCAN_LENGTH].lower()
        for pattern in _ERROR_PATTERNS:
            if pattern in response_start:
                return False

        return True

    async def route(self, subtask: Subtask, mode: Mode | None = None) -> RouteResult:
        """Find the best provider for *subtask* and execute the call."""
        mode = mode or self._config.mode
        is_search = subtask.type == TaskType.WEB_SEARCH
        use_quality_gate = mode in _QUALITY_GATE_MODES
        tokens_est = estimate_tokens(subtask.content)
        escalations: list[EscalationRecord] = []

        # Build scored candidate list
        candidates: list[tuple[str, float]] = []
        for name, provider in self._registry.get_all().items():
            if is_search and not isinstance(provider, SearchProvider):
                continue
            if not is_search and isinstance(provider, SearchProvider):
                continue
            if not self._registry.circuit_breaker.is_healthy(name):
                logger.info(f"Skipping {name} — circuit breaker open")
                continue

            pconfig = self._config.providers.get(name)
            ptype = pconfig.type if pconfig else ProviderType.FREE
            candidates.append((name, self.score(name, ptype, subtask, mode)))

        candidates.sort(key=lambda c: c[1], reverse=True)

        if not candidates:
            hint = _no_provider_hint(subtask.type)
            return RouteResult(
                type=subtask.type,
                response=f"No provider available for [{subtask.type.value}]. {hint}",
                provider="none",
                termination=TerminationState.NO_PROVIDER,
                estimated_tokens=tokens_est,
            )

        # Try candidates in score order, with quality gate escalation
        last_response: str | None = None
        last_response_provider: str | None = None
        last_response_score: float = 0.0
        failed_provider: str | None = None
        failed_reason: str | None = None
        for provider_name, provider_score in candidates:
            # Record escalation from previous failed provider to this one
            if failed_provider is not None:
                escalations.append(
                    EscalationRecord(
                        from_provider=failed_provider,
                        to_provider=provider_name,
                        reason=failed_reason or "unknown",
                    )
                )
                failed_provider = None
                failed_reason = None

            try:
                logger.info(f"Routing [{subtask.type}] -> {provider_name} (score={provider_score:.3f})")

                if is_search:
                    response = await self._registry.call_search(provider_name, subtask.content)
                else:
                    response = await self._registry.call_llm(provider_name, subtask.content)

                prov_cfg = self._config.providers.get(provider_name)
                is_paid = prov_cfg.type == ProviderType.PAID if prov_cfg else False
                self._quota.record_usage(
                    provider_name,
                    subtask.type.value,
                    is_paid=is_paid,
                    prompt=subtask.content,
                )

                # Quality gate: if response is low-quality, try next provider
                if use_quality_gate and not self.passes_quality_gate(response, subtask):
                    logger.warning(f"  {provider_name} failed quality gate, escalating")
                    last_response = response
                    last_response_provider = provider_name
                    last_response_score = provider_score
                    failed_provider = provider_name
                    failed_reason = "quality_gate"
                    continue

                self._registry.circuit_breaker.record_success(provider_name)
                termination = TerminationState.ESCALATED if escalations else TerminationState.COMPLETED
                return RouteResult(
                    type=subtask.type,
                    response=response,
                    provider=provider_name,
                    score=provider_score,
                    termination=termination,
                    escalations=escalations,
                    estimated_tokens=tokens_est,
                )
            except Exception as e:
                logger.warning(f"  {provider_name} failed: {type(e).__name__}")
                self._registry.circuit_breaker.record_failure(provider_name)
                failed_provider = provider_name
                failed_reason = "provider_error"

        # If we have a low-quality response, return it rather than nothing
        if last_response is not None and last_response_provider is not None:
            logger.warning("All providers failed quality gate, returning best available")
            return RouteResult(
                type=subtask.type,
                response=last_response,
                provider=last_response_provider,
                score=last_response_score,
                termination=TerminationState.QUALITY_GATE_FALLBACK,
                escalations=escalations,
                estimated_tokens=tokens_est,
            )

        return RouteResult(
            type=subtask.type,
            response=f"All providers failed for [{subtask.type.value}]. Check your API keys and provider status.",
            provider="none",
            termination=TerminationState.ALL_FAILED,
            escalations=escalations,
            estimated_tokens=tokens_est,
        )
