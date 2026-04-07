"""Router — scores providers and routes each subtask to the best one.

Scoring formula (additive weighted):
  Score = w_quality × Quality + w_cost × CostScore + w_availability × AvailabilityScore

Weights vary by mode (economy favors cost, quality favors quality).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from smartsplit.exceptions import ProviderError
from smartsplit.models import (
    Complexity,
    EscalationRecord,
    Mode,
    ProviderType,
    RouteResult,
    Subtask,
    TaskType,
    TerminationState,
    TokenUsage,
)
from smartsplit.providers.base import SearchProvider
from smartsplit.quota import estimate_tokens

if TYPE_CHECKING:
    from smartsplit.config import ProviderConfig, SmartSplitConfig
    from smartsplit.learning import BanditScorer
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

# Default competence score for providers not in the competence table
_DEFAULT_COMPETENCE_SCORE = 5

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
    # Refusals — based on llm-attacks (academic reference) + CHI 2024 research
    "i cannot",
    "i can't",
    "i am sorry",
    "i'm sorry",
    "i apologize",
    "i'm unable",
    "i am unable",
    "i'm not able",
    "i am not able",
    "i'm just",
    "i'm an ai",
    "i do not",
    "i don't have",
    "as an ai",
    "as a language model",
    "as an assistant",
    "my apologies",
    "i must respectfully",
    "it is not appropriate",
    "it's not appropriate",
    # Refusals (French)
    "je ne peux pas",
    "je suis désolé",
    "je m'excuse",
    "en tant qu'ia",
    "en tant que modèle",
    # Refusals (Spanish)
    "no puedo",
    "lo siento",
    "como modelo de lenguaje",
    # Refusals (German)
    "ich kann nicht",
    "als ki",
    "es tut mir leid",
    # Rate limits / errors (language-agnostic)
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

# LLM refusal check — only triggered for suspicious short responses
_MIN_LLM_CHECK_LEN = 50  # below this, length check already catches it
_MAX_LLM_CHECK_LEN = 300  # above this, response is likely substantive


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


def _resolve_tier(pconfig: ProviderConfig | None, complexity: Complexity, mode: Mode = Mode.BALANCED) -> str:
    """Determine model tier ('fast' or 'strong') based on provider config, complexity, and mode.

    Free providers always return 'fast' (single model).
    Paid providers use 'strong' for:
      - high complexity tasks (any mode)
      - medium complexity tasks in quality mode
    """
    if pconfig is None or pconfig.type == ProviderType.FREE:
        return "fast"
    if not pconfig.strong_model:
        return "fast"
    if complexity == Complexity.HIGH:
        return "strong"
    if complexity == Complexity.MEDIUM and mode == Mode.QUALITY:
        return "strong"
    return "fast"


def _get_model_for_tier(pconfig: ProviderConfig | None, tier: str) -> str | None:
    """Get the model name for a given tier. Returns None for free providers (use default)."""
    if pconfig is None or pconfig.type == ProviderType.FREE:
        return None
    if tier == "strong" and pconfig.strong_model:
        return pconfig.strong_model
    if pconfig.fast_model:
        return pconfig.fast_model
    return None


class Router:
    """Routes ``Subtask`` objects to the best available provider."""

    def __init__(
        self,
        registry: ProviderRegistry,
        quota: QuotaTracker,
        config: SmartSplitConfig,
        bandit: BanditScorer | None = None,
    ) -> None:
        self._registry = registry
        self._quota = quota
        self._config = config
        self._bandit = bandit

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

        # Quality: MAB score (learned) with competence table as prior (static)
        competence = self._config.competence_table.get(
            subtask.type.value,
            self._config.competence_table.get("general", {}),
        )
        pconfig = self._config.providers.get(provider_name)
        tier = _resolve_tier(pconfig, subtask.complexity)
        tiered_key = f"{provider_name}:{tier}"
        static_quality = competence.get(tiered_key, competence.get(provider_name, _DEFAULT_COMPETENCE_SCORE)) / 10.0

        # Use tiered key for MAB so fast/strong are scored independently
        bandit_key = tiered_key if pconfig and pconfig.type == ProviderType.PAID else provider_name
        if self._bandit:
            quality = self._bandit.score(subtask.type.value, bandit_key, prior=static_quality)
        else:
            quality = static_quality

        # Cost: free = 1.0, paid:fast = 0.3 (cheap paid), paid:strong = 0.0 (expensive)
        if provider_type == ProviderType.FREE:
            cost_score = 1.0
        elif tier == "fast":
            cost_score = 0.3
        else:
            cost_score = 0.0

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

        Checks (language-agnostic where possible):
        1. Empty / too short
        2. Refusal patterns (beginning of response)
        3. Substance check — repetition ratio, word diversity
        4. Structure check — code presence when code is requested
        """
        if not response or not response.strip():
            return False

        stripped = response.strip()

        # 1. Length check based on complexity
        min_len = _MIN_RESPONSE_LENGTH.get(subtask.complexity, 5)
        if len(stripped) < min_len:
            return False

        # 2. Refusal pattern check — scan beginning only
        response_start = stripped[:_ERROR_SCAN_LENGTH].lower()
        for pattern in _ERROR_PATTERNS:
            if pattern in response_start:
                return False

        # 3. Substance checks (language-agnostic)
        if len(stripped) > 50:
            words = stripped.split()
            if words:
                # High repetition = low quality (e.g. "the the the the")
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.15:
                    return False

                # Very short average word length = gibberish
                avg_word_len = sum(len(w) for w in words) / len(words)
                if avg_word_len < 1.5:
                    return False

            # Response is mostly whitespace / newlines
            content_ratio = len(stripped.replace(" ", "").replace("\n", "")) / len(stripped)
            if content_ratio < 0.3:
                return False

        # 4. Structure check — code tasks with "write"/"implement" should contain code
        if subtask.type == TaskType.CODE and subtask.complexity != Complexity.LOW:
            prompt_lower = subtask.content[:200].lower()
            asks_for_code = any(w in prompt_lower for w in ("write", "implement", "create", "build", "code"))
            if asks_for_code:
                has_code = "```" in response or "def " in response or "class " in response or "function " in response
                if not has_code and len(stripped) > 200:
                    return False

        return True

    async def _check_refusal_llm(self, response: str) -> bool:
        """Use a free LLM to check if a response is a refusal (multilingual).

        Returns True if the response IS a refusal. Only called for suspicious
        short responses that passed pattern matching. Result feeds the MAB.
        """
        try:
            result = await self._registry.call_free_llm(
                "Is this AI response a refusal or apology for not being able to help? "
                "Answer ONLY 'yes' or 'no'.\n\n"
                f"Response: {response[:300]}",
                prefer="groq",
            )
            return result.strip().lower().startswith("yes")
        except Exception:
            return False

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
        model: str | None = None
        prov_cfg = None
        messages: list[dict[str, str]] | None = None
        bandit_key: str = ""
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

                usage = TokenUsage()
                if is_search:
                    bandit_key = provider_name
                    response = await self._registry.call_search(provider_name, subtask.content)
                else:
                    prov_cfg = self._config.providers.get(provider_name)
                    tier = _resolve_tier(prov_cfg, subtask.complexity, mode)
                    bandit_key = (
                        f"{provider_name}:{tier}" if prov_cfg and prov_cfg.type == ProviderType.PAID else provider_name
                    )
                    model = _get_model_for_tier(prov_cfg, tier)
                    if model:
                        logger.info(f"  Using {tier} model: {model}")
                    # If subtask has conversation messages, update the last user message
                    # with the enriched content (context injection, enrichment, etc.)
                    messages = subtask.messages
                    if messages:
                        messages = [dict(m) for m in messages]  # shallow copy
                        for i in range(len(messages) - 1, -1, -1):
                            if messages[i]["role"] == "user":
                                messages[i] = {"role": "user", "content": subtask.content}
                                break
                    response, usage = await self._registry.call_llm(
                        provider_name, subtask.content, model=model, messages=messages
                    )

                logger.debug(f"  {provider_name} response: {response[:150]!r}")

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
                    if self._bandit:
                        self._bandit.record(subtask.type.value, bandit_key, success=False)
                    # Keep the first (highest-scored) response as fallback
                    if last_response is None:
                        last_response = response
                        last_response_provider = provider_name
                        last_response_score = provider_score
                    failed_provider = provider_name
                    failed_reason = "quality_gate"
                    continue

                # LLM refusal check — only for suspicious short responses (50-300 chars)
                # that passed pattern matching. Catches multilingual refusals.
                resp_len = len(response.strip())
                if use_quality_gate and _MIN_LLM_CHECK_LEN <= resp_len <= _MAX_LLM_CHECK_LEN:
                    is_refusal = await self._check_refusal_llm(response)
                    if is_refusal:
                        logger.warning(f"  {provider_name} LLM detected refusal, escalating")
                        if self._bandit:
                            self._bandit.record(subtask.type.value, bandit_key, success=False)
                        if last_response is None:
                            last_response = response
                            last_response_provider = provider_name
                            last_response_score = provider_score
                        failed_provider = provider_name
                        failed_reason = "llm_refusal_check"
                        continue

                self._registry.circuit_breaker.record_success(provider_name)
                if self._bandit:
                    self._bandit.record(subtask.type.value, bandit_key, success=True)
                termination = TerminationState.ESCALATED if escalations else TerminationState.COMPLETED
                return RouteResult(
                    type=subtask.type,
                    response=response,
                    provider=provider_name,
                    score=provider_score,
                    termination=termination,
                    escalations=escalations,
                    estimated_tokens=tokens_est,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )
            except (ProviderError, httpx.HTTPError, TimeoutError) as e:
                error_msg = str(e).lower()
                is_context_error = any(
                    p in error_msg
                    for p in ("context length", "context_length", "max_tokens", "too long", "token limit")
                )
                if is_context_error:
                    # Context too long — not the provider's fault, don't penalize
                    logger.info(f"  {provider_name} context too long, trying next provider")
                    failed_provider = provider_name
                    failed_reason = "context_too_long"
                    continue
                logger.warning(f"  {provider_name} failed: {type(e).__name__}")
                # If strong model failed, retry with fast model on same provider
                if not is_search and model and prov_cfg and prov_cfg.fast_model and model != prov_cfg.fast_model:
                    try:
                        logger.info(f"  Retrying {provider_name} with fast model: {prov_cfg.fast_model}")
                        response, fast_usage = await self._registry.call_llm(
                            provider_name,
                            subtask.content,
                            model=prov_cfg.fast_model,
                            messages=messages,
                        )
                        self._registry.circuit_breaker.record_success(provider_name)
                        self._quota.record_usage(
                            provider_name,
                            subtask.type.value,
                            is_paid=prov_cfg.type == ProviderType.PAID,
                            prompt=subtask.content,
                        )
                        termination = TerminationState.ESCALATED if escalations else TerminationState.COMPLETED
                        return RouteResult(
                            type=subtask.type,
                            response=response,
                            provider=provider_name,
                            score=provider_score,
                            termination=termination,
                            escalations=escalations,
                            estimated_tokens=tokens_est,
                            prompt_tokens=fast_usage.prompt_tokens,
                            completion_tokens=fast_usage.completion_tokens,
                        )
                    except (ProviderError, httpx.HTTPError, TimeoutError) as e2:
                        logger.warning(f"  {provider_name} fast model also failed: {type(e2).__name__}")
                self._registry.circuit_breaker.record_failure(provider_name)
                if self._bandit:
                    self._bandit.record(subtask.type.value, bandit_key, success=False)
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
