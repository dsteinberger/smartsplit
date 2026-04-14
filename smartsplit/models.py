"""Domain models for SmartSplit — single source of truth for all data structures."""

from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


def short_id() -> str:
    """Generate a short 8-char request ID."""
    return uuid.uuid4().hex[:8]


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TaskType(StrEnum):
    WEB_SEARCH = "web_search"
    SUMMARIZE = "summarize"
    CODE = "code"
    REASONING = "reasoning"
    TRANSLATION = "translation"
    BOILERPLATE = "boilerplate"
    GENERAL = "general"
    MATH = "math"
    CREATIVE = "creative"
    FACTUAL = "factual"
    EXTRACTION = "extraction"


class Complexity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Mode(StrEnum):
    ECONOMY = "economy"
    BALANCED = "balanced"
    QUALITY = "quality"


class ProviderType(StrEnum):
    FREE = "free"
    PAID = "paid"


class ContextTier(StrEnum):
    """Worker context tier — controls how much prompt context is sent to a provider."""

    SMALL = "small"  # ~1k tokens / 4k chars — tight free tiers (Cerebras, Groq)
    MEDIUM = "medium"  # ~4k tokens / 16k chars — generous free tiers (Mistral, OpenRouter)
    LARGE = "large"  # ~16k tokens / 64k chars — paid or high-limit (Gemini, paid APIs)


class Subtask(BaseModel):
    type: TaskType = TaskType.GENERAL
    content: str
    complexity: Complexity = Complexity.MEDIUM
    depends_on: int | None = Field(
        default=None,
        description="Index (0-based) of another subtask this one depends on. None = independent.",
    )
    messages: list[dict[str, str]] | None = Field(
        default=None,
        description="Original conversation messages. Set for single-subtask prompts to preserve context.",
        exclude=True,
    )


class TerminationState(StrEnum):
    COMPLETED = "completed"
    ESCALATED = "escalated"
    QUALITY_GATE_FALLBACK = "quality_gate_fallback"
    ALL_FAILED = "all_failed"
    NO_PROVIDER = "no_provider"


class EscalationRecord(BaseModel):
    """Records one escalation event during routing."""

    model_config = ConfigDict(frozen=True)

    from_provider: str
    to_provider: str
    reason: str  # "quality_gate", "provider_error", "rate_limit"


class RouteResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: TaskType
    response: str
    provider: str
    score: float = 0.0
    termination: TerminationState = TerminationState.COMPLETED
    escalations: list[EscalationRecord] = Field(default_factory=list)
    estimated_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class SavingsReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_requests: int
    free_requests: int
    paid_requests: int
    free_percentage: float
    estimated_tokens_saved: int
    estimated_cost_saved_usd: float
    providers_usage: dict[str, int] = Field(default_factory=dict)


class ProviderStatus(BaseModel):
    name: str
    type: ProviderType
    enabled: bool
    availability: float = Field(ge=0.0, le=1.0, description="Remaining quota ratio")
    requests_today: int = 0


class RoutingStep(BaseModel):
    """One subtask routing decision — used for observability."""

    task_type: TaskType
    complexity: Complexity = Complexity.MEDIUM
    provider: str
    score: float = 0.0
    is_paid: bool = False
    termination: TerminationState = TerminationState.COMPLETED
    escalations: list[EscalationRecord] = Field(default_factory=list)
    estimated_tokens: int = 0


class RequestLog(BaseModel):
    """Full trace of a smart_query request."""

    request_id: str = Field(default_factory=short_id)
    timestamp: str
    prompt_preview: str = Field(description="First 100 chars of the prompt")
    mode: Mode
    domains_detected: list[str] = Field(default_factory=list)
    subtask_count: int
    steps: list[RoutingStep] = Field(default_factory=list)
    free_calls: int = 0
    paid_calls: int = 0
    tokens_saved: int = 0
