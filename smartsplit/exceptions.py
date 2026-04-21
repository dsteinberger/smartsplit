"""Custom exceptions for SmartSplit."""

from __future__ import annotations


class SmartSplitError(Exception):
    """Base exception for all SmartSplit errors."""


class ProviderError(SmartSplitError):
    """A provider call failed."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class ProviderAuthError(ProviderError):
    """Provider rejected the API key (HTTP 401/403).

    Signals a configuration problem — retrying does not help. The registry
    trips the circuit breaker immediately on this error.
    """


class ProviderRateLimitError(ProviderError):
    """Provider rate limit exceeded (HTTP 429).

    Carries the ``retry_after`` hint from the response (if any) so callers
    can skip the provider for a bounded time instead of tripping the
    breaker: a 429 means "slow down", not "I am down".
    """

    def __init__(self, provider: str, message: str, retry_after: float | None = None) -> None:
        super().__init__(provider, message)
        self.retry_after = retry_after


class NoProviderAvailableError(SmartSplitError):
    """No provider could handle the request."""

    def __init__(self, task_type: str):
        self.task_type = task_type
        super().__init__(f"No provider available for task type: {task_type}")


class ConfigError(SmartSplitError):
    """Invalid or missing configuration."""
