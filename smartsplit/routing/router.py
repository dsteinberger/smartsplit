"""Router — scores providers and routes each subtask to the best one.

Scoring formula (additive weighted):
  Score = w_quality × Quality + w_cost × CostScore + w_availability × AvailabilityScore

Weights vary by mode (economy favors cost, quality favors quality).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
from smartsplit.routing.quota import estimate_tokens

_CONTEXT_ERROR_MARKERS = ("context length", "context_length", "max_tokens", "too long", "token limit")


@dataclass
class _RouteState:
    """Mutable state accumulated across provider attempts within a single route() call."""

    subtask: Subtask
    mode: Mode
    is_search: bool
    use_quality_gate: bool
    tokens_est: int
    escalations: list[EscalationRecord] = field(default_factory=list)
    fallback: tuple[str, str, float] | None = None  # (response, provider, score)
    pending_failure: tuple[str, str] | None = None  # (provider, reason)


@dataclass(frozen=True)
class _CallContext:
    """Resolved parameters for a single provider call."""

    bandit_key: str
    tier: str
    model: str | None
    pconfig: ProviderConfig | None
    messages: list[dict[str, str]] | None


if TYPE_CHECKING:
    from smartsplit.config import ProviderConfig, SmartSplitConfig
    from smartsplit.providers.registry import ProviderRegistry
    from smartsplit.routing.learning import BanditScorer
    from smartsplit.routing.quota import QuotaTracker

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
# Kept short so "I apologize for the confusion. Here's the code: ..." is not rejected.
_ERROR_SCAN_LENGTH = 80

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

    @staticmethod
    def _compute_weights(mode: Mode, complexity: Complexity) -> tuple[float, float, float]:
        """Return (w_quality, w_cost, w_availability) weights for ``mode`` + ``complexity``."""
        w_quality, w_cost, w_avail = _MODE_WEIGHTS.get(mode, _MODE_WEIGHTS[Mode.BALANCED])
        boost = _COMPLEXITY_QUALITY_BOOST.get(complexity, 0.0)
        return (
            min(1.0, max(0.0, w_quality + boost)),
            min(1.0, max(0.0, w_cost - boost)),
            w_avail,
        )

    def _compute_quality(
        self, provider_name: str, subtask: Subtask, tier: str, pconfig: ProviderConfig | None
    ) -> float:
        """Blend the static competence score with the MAB's learned score for this tier.

        Lookup order for the static score:
          1. ``competence_table[f"{type}.{domain}"][provider]`` when ``subtask.domain`` is set
          2. ``competence_table[type][provider]`` (generic task-type score)
          3. ``competence_table["general"][provider]`` (cross-task fallback)

        The bandit key stays on ``type`` (no ``.domain`` suffix) so learning
        converges faster — the domain-specific prior already biases the score.
        """
        tiered_key = f"{provider_name}:{tier}"

        def _lookup(table_key: str) -> int | None:
            row = self._config.competence_table.get(table_key)
            if not row:
                return None
            return row.get(tiered_key, row.get(provider_name))

        static_score: int | None = None
        if subtask.domain:
            static_score = _lookup(f"{subtask.type.value}.{subtask.domain}")
        if static_score is None:
            static_score = _lookup(subtask.type.value)
        if static_score is None:
            static_score = _lookup("general")
        if static_score is None:
            static_score = _DEFAULT_COMPETENCE_SCORE

        static_quality = static_score / 10.0
        bandit_key = tiered_key if pconfig and pconfig.type == ProviderType.PAID else provider_name
        if self._bandit:
            return self._bandit.score(subtask.type.value, bandit_key, prior=static_quality)
        return static_quality

    @staticmethod
    def _compute_cost(provider_type: ProviderType, tier: str) -> float:
        """Cost score: free = 1.0, paid fast = 0.5, paid strong = 0.2."""
        if provider_type == ProviderType.FREE:
            return 1.0
        if tier == "fast":
            return 0.5
        return 0.2

    def _compute_availability(self, provider_name: str) -> float:
        """1.0 when ample quota remains, degrades linearly below the danger threshold."""
        raw_avail = self._quota.get_availability(provider_name)
        if raw_avail >= _AVAILABILITY_DANGER_THRESHOLD:
            return 1.0
        if raw_avail <= 0.0:
            return 0.0
        return raw_avail / _AVAILABILITY_DANGER_THRESHOLD

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
        w_quality, w_cost, w_avail = self._compute_weights(mode, subtask.complexity)
        pconfig = self._config.providers.get(provider_name)
        tier = _resolve_tier(pconfig, subtask.complexity)

        quality = self._compute_quality(provider_name, subtask, tier, pconfig)
        cost_score = self._compute_cost(provider_type, tier)
        avail_score = self._compute_availability(provider_name)

        return w_quality * quality + w_cost * cost_score + w_avail * avail_score

    @staticmethod
    def _reject_on_length(stripped: str, complexity: Complexity) -> str | None:
        """Reject responses shorter than the minimum for the given complexity."""
        min_len = _MIN_RESPONSE_LENGTH.get(complexity, 5)
        if len(stripped) < min_len:
            return f"too_short ({len(stripped)}<{min_len})"
        return None

    @staticmethod
    def _reject_on_refusal(stripped: str) -> str | None:
        """Reject refusals/boilerplate apologies detected at the start of the response."""
        response_start = stripped[:_ERROR_SCAN_LENGTH].lower()
        for pattern in _ERROR_PATTERNS:
            if pattern in response_start:
                return f"refusal_pattern ({pattern!r})"
        return None

    @staticmethod
    def _reject_on_substance(stripped: str) -> str | None:
        """Reject low-substance responses: heavy repetition, gibberish, or mostly whitespace."""
        if len(stripped) <= 50:
            return None
        words = stripped.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return f"repetition ({unique_ratio:.0%} unique)"
            avg_word_len = sum(len(w) for w in words) / len(words)
            if avg_word_len < 1.5:
                return f"gibberish (avg_word_len={avg_word_len:.1f})"
        content_ratio = len(stripped.replace(" ", "").replace("\n", "")) / len(stripped)
        if content_ratio < 0.3:
            return f"mostly_whitespace ({content_ratio:.0%} content)"
        return None

    @staticmethod
    def _reject_on_missing_code(stripped: str, response: str, subtask: Subtask) -> str | None:
        """Reject long code-task responses that ship no code snippet."""
        if subtask.type != TaskType.CODE or subtask.complexity == Complexity.LOW:
            return None
        prompt_lower = subtask.content[:200].lower()
        if not any(w in prompt_lower for w in ("write", "implement", "create", "build", "code")):
            return None
        has_code = "```" in response or "def " in response or "class " in response or "function " in response
        if not has_code and len(stripped) > 200:
            return "missing_code"
        return None

    @staticmethod
    def quality_gate_reason(response: str, subtask: Subtask) -> str | None:
        """Return a short reason why the response fails the quality gate, or None if it passes.

        Checks (language-agnostic where possible):
        1. Empty / too short
        2. Refusal patterns (beginning of response)
        3. Substance — repetition ratio, word diversity, whitespace density
        4. Structure — code presence when code is requested
        """
        if not response or not response.strip():
            return "empty"
        stripped = response.strip()

        for check in (
            Router._reject_on_length(stripped, subtask.complexity),
            Router._reject_on_refusal(stripped),
            Router._reject_on_substance(stripped),
            Router._reject_on_missing_code(stripped, response, subtask),
        ):
            if check is not None:
                return check
        return None

    @staticmethod
    def passes_quality_gate(response: str, subtask: Subtask) -> bool:
        """Check if a response meets minimum quality standards."""
        return Router.quality_gate_reason(response, subtask) is None

    async def _check_refusal_llm(self, response: str) -> bool:
        """Use a worker LLM to check if a response is a refusal (multilingual).

        Returns True if the response IS a refusal. Only called for suspicious
        short responses that passed pattern matching. Result feeds the MAB.
        """
        try:
            result = await self._registry.call_worker_llm(
                "Is this AI response a refusal or apology for not being able to help? "
                "Answer ONLY 'yes' or 'no'.\n\n"
                f"Response: {response[:300]}",
                prefer="groq",
            )
            return result.strip().lower().startswith("yes")
        except Exception:
            return False

    def _resolve_override(self, subtask_type: str) -> str | None:
        """Return an override provider name if it's configured and healthy, else ``None``."""
        name = self._config.overrides.get(subtask_type)
        if not name:
            return None
        if self._registry.get(name) is None:
            logger.warning("Override %s → %s ignored — provider not configured", subtask_type, name)
            return None
        if not self._registry.circuit_breaker.is_healthy(name):
            logger.warning("Override %s → %s skipped — circuit breaker open", subtask_type, name)
            return None
        return name

    def _build_candidates(self, subtask: Subtask, mode: Mode, is_search: bool) -> list[tuple[str, float]]:
        """Score every healthy provider compatible with ``subtask`` and sort descending."""
        candidates: list[tuple[str, float]] = []
        for name, provider in self._registry.get_all().items():
            if is_search and not isinstance(provider, SearchProvider):
                continue
            if not is_search and isinstance(provider, SearchProvider):
                continue
            if not self._registry.circuit_breaker.is_healthy(name):
                logger.info("Skipping %s — circuit breaker open", name)
                continue
            pconfig = self._config.providers.get(name)
            ptype = pconfig.type if pconfig else ProviderType.FREE
            candidates.append((name, self.score(name, ptype, subtask, mode)))
        candidates.sort(key=lambda c: c[1], reverse=True)
        return candidates

    def _build_messages_for_call(self, subtask: Subtask) -> list[dict[str, str]] | None:
        """Shallow-copy ``subtask.messages`` and overwrite the last user content.

        The enriched content (context injection, search results) lives in ``subtask.content``.
        """
        if not subtask.messages:
            return None
        messages = [dict(m) for m in subtask.messages]
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i] = {"role": "user", "content": subtask.content}
                break
        return messages

    def _prepare_call_context(self, provider_name: str, subtask: Subtask, mode: Mode) -> _CallContext:
        """Resolve tier, model, bandit key, and prepared messages for an LLM call."""
        pconfig = self._config.providers.get(provider_name)
        tier = _resolve_tier(pconfig, subtask.complexity, mode)
        bandit_key = f"{provider_name}:{tier}" if pconfig and pconfig.type == ProviderType.PAID else provider_name
        model = _get_model_for_tier(pconfig, tier)
        return _CallContext(
            bandit_key=bandit_key,
            tier=tier,
            model=model,
            pconfig=pconfig,
            messages=self._build_messages_for_call(subtask),
        )

    async def _execute_call(
        self,
        provider_name: str,
        subtask: Subtask,
        state: _RouteState,
        call_ctx: _CallContext,
    ) -> tuple[str, TokenUsage]:
        """Execute a search or LLM call using the prepared ``call_ctx``."""
        if state.is_search:
            response = await self._registry.call_search(provider_name, subtask.content)
            return response, TokenUsage()
        if call_ctx.model:
            logger.info("  Using %s model: %s", call_ctx.tier, call_ctx.model)
        response, usage = await self._registry.call_llm(
            provider_name, subtask.content, model=call_ctx.model, messages=call_ctx.messages
        )
        return response, usage

    def _build_call_ctx_for_state(self, provider_name: str, subtask: Subtask, state: _RouteState) -> _CallContext:
        """Pick between search and LLM call context depending on state."""
        if state.is_search:
            return _CallContext(bandit_key=provider_name, tier="search", model=None, pconfig=None, messages=None)
        return self._prepare_call_context(provider_name, subtask, state.mode)

    def _record_quota(self, provider_name: str, subtask: Subtask, pconfig: ProviderConfig | None) -> None:
        is_paid = pconfig.type == ProviderType.PAID if pconfig else False
        self._quota.record_usage(provider_name, subtask.type.value, is_paid=is_paid, prompt=subtask.content)

    def _remember_fallback(self, state: _RouteState, provider_name: str, response: str, score: float) -> None:
        """Keep the first (highest-scored) response as the quality-gate fallback."""
        if state.fallback is None:
            state.fallback = (response, provider_name, score)

    def _mark_failure(self, state: _RouteState, provider_name: str, reason: str) -> None:
        state.pending_failure = (provider_name, reason)

    async def _passes_refusal_check(self, response: str, use_quality_gate: bool) -> bool:
        """Return True if no refusal is detected — cheap short-circuits first."""
        if not use_quality_gate:
            return True
        stripped_len = len(response.strip())
        if not (_MIN_LLM_CHECK_LEN <= stripped_len <= _MAX_LLM_CHECK_LEN):
            return True
        return not await self._check_refusal_llm(response)

    async def _try_fast_model_retry(
        self,
        provider_name: str,
        provider_score: float,
        subtask: Subtask,
        state: _RouteState,
        call_ctx: _CallContext,
    ) -> RouteResult | None:
        """Retry the same provider with its fast model after a strong-model failure."""
        pcfg = call_ctx.pconfig
        if state.is_search or not call_ctx.model or pcfg is None or not pcfg.fast_model:
            return None
        if call_ctx.model == pcfg.fast_model:
            return None
        try:
            logger.info("  Retrying %s with fast model: %s", provider_name, pcfg.fast_model)
            response, fast_usage = await self._registry.call_llm(
                provider_name, subtask.content, model=pcfg.fast_model, messages=call_ctx.messages
            )
        except (ProviderError, httpx.HTTPError, TimeoutError) as exc:
            logger.warning("  %s fast model also failed: %s", provider_name, type(exc).__name__)
            return None

        self._registry.circuit_breaker.record_success(provider_name)
        if self._bandit:
            self._bandit.record(subtask.type.value, f"{provider_name}:fast", success=True)
        self._record_quota(provider_name, subtask, pcfg)
        return self._build_success_result(subtask, state, provider_name, provider_score, response, fast_usage)

    def _build_success_result(
        self,
        subtask: Subtask,
        state: _RouteState,
        provider_name: str,
        provider_score: float,
        response: str,
        usage: TokenUsage,
    ) -> RouteResult:
        termination = TerminationState.ESCALATED if state.escalations else TerminationState.COMPLETED
        return RouteResult(
            type=subtask.type,
            response=response,
            provider=provider_name,
            score=provider_score,
            termination=termination,
            escalations=list(state.escalations),
            estimated_tokens=state.tokens_est,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )

    @staticmethod
    def _is_context_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(p in msg for p in _CONTEXT_ERROR_MARKERS)

    async def _attempt_one(
        self,
        provider_name: str,
        provider_score: float,
        subtask: Subtask,
        state: _RouteState,
    ) -> RouteResult | None:
        """Try a single provider. Returns a RouteResult on success, else mutates state and returns None."""
        logger.info("Routing [%s] -> %s (score=%.3f)", subtask.type, provider_name, provider_score)
        call_ctx = self._build_call_ctx_for_state(provider_name, subtask, state)

        try:
            response, usage = await self._execute_call(provider_name, subtask, state, call_ctx)
            logger.debug("  %s response: %r", provider_name, response[:150])
            self._record_quota(provider_name, subtask, call_ctx.pconfig)
        except (ProviderError, httpx.HTTPError, TimeoutError) as exc:
            if self._is_context_error(exc):
                logger.info("  %s context too long, trying next provider", provider_name)
                self._mark_failure(state, provider_name, "context_too_long")
                return None
            logger.warning("  %s failed: %s", provider_name, type(exc).__name__)
            retry = await self._try_fast_model_retry(provider_name, provider_score, subtask, state, call_ctx)
            if retry is not None:
                return retry
            self._registry.circuit_breaker.record_failure(provider_name)
            if self._bandit:
                self._bandit.record(subtask.type.value, call_ctx.bandit_key, success=False)
            self._mark_failure(state, provider_name, "provider_error")
            return None

        if state.use_quality_gate:
            gate_reject = self.quality_gate_reason(response, subtask)
            if gate_reject:
                logger.warning("  %s failed quality gate: %s, escalating", provider_name, gate_reject)
                if self._bandit:
                    self._bandit.record(subtask.type.value, call_ctx.bandit_key, success=False)
                self._remember_fallback(state, provider_name, response, provider_score)
                self._mark_failure(state, provider_name, "quality_gate")
                return None

        if not await self._passes_refusal_check(response, state.use_quality_gate):
            logger.warning("  %s LLM detected refusal, escalating", provider_name)
            if self._bandit:
                self._bandit.record(subtask.type.value, call_ctx.bandit_key, success=False)
            self._remember_fallback(state, provider_name, response, provider_score)
            self._mark_failure(state, provider_name, "llm_refusal_check")
            return None

        self._registry.circuit_breaker.record_success(provider_name)
        if self._bandit:
            self._bandit.record(subtask.type.value, call_ctx.bandit_key, success=True)
        return self._build_success_result(subtask, state, provider_name, provider_score, response, usage)

    async def route(self, subtask: Subtask, mode: Mode | None = None) -> RouteResult:
        """Find the best provider for *subtask* and execute the call."""
        effective_mode = mode or self._config.mode
        state = _RouteState(
            subtask=subtask,
            mode=effective_mode,
            is_search=subtask.type == TaskType.WEB_SEARCH,
            use_quality_gate=effective_mode in _QUALITY_GATE_MODES,
            tokens_est=estimate_tokens(subtask.content),
        )

        override_name = self._resolve_override(subtask.type.value)
        candidates = self._build_candidates(subtask, effective_mode, state.is_search)

        if candidates and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Scores for [%s] (%s): %s",
                subtask.type.value,
                effective_mode,
                ", ".join(f"{n}={s:.3f}" for n, s in candidates),
            )

        if override_name:
            candidates = [(n, s) for n, s in candidates if n != override_name]
            candidates.insert(0, (override_name, 10.0))
            logger.info("Override active: %s → %s", subtask.type.value, override_name)

        if not candidates:
            hint = _no_provider_hint(subtask.type)
            return RouteResult(
                type=subtask.type,
                response=f"No provider available for [{subtask.type.value}]. {hint}",
                provider="none",
                termination=TerminationState.NO_PROVIDER,
                estimated_tokens=state.tokens_est,
            )

        for provider_name, provider_score in candidates:
            if state.pending_failure is not None:
                failed_provider, failed_reason = state.pending_failure
                state.escalations.append(
                    EscalationRecord(from_provider=failed_provider, to_provider=provider_name, reason=failed_reason)
                )
                state.pending_failure = None

            result = await self._attempt_one(provider_name, provider_score, subtask, state)
            if result is not None:
                return result

        if state.fallback is not None:
            logger.warning("All providers failed quality gate, returning best available")
            response, provider, score = state.fallback
            return RouteResult(
                type=subtask.type,
                response=response,
                provider=provider,
                score=score,
                termination=TerminationState.QUALITY_GATE_FALLBACK,
                escalations=list(state.escalations),
                estimated_tokens=state.tokens_est,
            )

        return RouteResult(
            type=subtask.type,
            response=f"All providers failed for [{subtask.type.value}]. Check your API keys and provider status.",
            provider="none",
            termination=TerminationState.ALL_FAILED,
            escalations=list(state.escalations),
            estimated_tokens=state.tokens_est,
        )
