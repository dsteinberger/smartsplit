"""Custom exceptions for SmartSplit."""

from __future__ import annotations


class SmartSplitError(Exception):
    """Base exception for all SmartSplit errors."""


class ProviderError(SmartSplitError):
    """A provider call failed."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class NoProviderAvailableError(SmartSplitError):
    """No provider could handle the request."""

    def __init__(self, task_type: str):
        self.task_type = task_type
        super().__init__(f"No provider available for task type: {task_type}")


class ConfigError(SmartSplitError):
    """Invalid or missing configuration."""
