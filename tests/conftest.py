"""Shared fixtures for SmartSplit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from smartsplit.config import ProviderConfig, SmartSplitConfig
from smartsplit.models import Mode, ProviderType
from smartsplit.providers.registry import ProviderRegistry
from smartsplit.quota import QuotaTracker

# ── Config fixtures ──────────────────────────────────────────


SAMPLE_COMPETENCE = {
    "web_search": {"serper": 9, "tavily": 9, "groq": 3, "gemini": 7, "anthropic": 5},
    "summarize": {"groq": 8, "gemini": 8, "anthropic": 9},
    "code": {"groq": 7, "gemini": 8, "deepseek": 9, "anthropic": 10},
    "reasoning": {"groq": 6, "gemini": 8, "deepseek": 8, "anthropic": 10},
    "translation": {"mistral": 9, "gemini": 8, "groq": 7},
    "general": {"groq": 7, "gemini": 8, "anthropic": 9},
    "math": {"groq": 5, "gemini": 8, "deepseek": 9, "anthropic": 10},
    "creative": {"groq": 5, "gemini": 7, "anthropic": 10},
    "factual": {"groq": 8, "gemini": 9, "anthropic": 8},
    "extraction": {"groq": 8, "gemini": 8, "anthropic": 9},
}


def _make_provider_config(
    name: str,
    api_key: str = "test_key",
    enabled: bool = True,
) -> ProviderConfig:
    ptype = ProviderType.PAID if name in ("anthropic", "openai") else ProviderType.FREE
    limits = {"rpd": 14400} if name == "groq" else {"rpd": 1500} if name == "gemini" else {}
    return ProviderConfig(api_key=api_key, type=ptype, enabled=enabled, limits=limits)


@pytest.fixture
def make_config():
    """Factory fixture — call with a list of provider names."""

    def _factory(providers: list[str], mode: Mode = Mode.BALANCED) -> SmartSplitConfig:
        return SmartSplitConfig(
            mode=mode,
            providers={p: _make_provider_config(p) for p in providers},
            competence_table=SAMPLE_COMPETENCE,
        )

    return _factory


# ── Quota fixture ────────────────────────────────────────────


@pytest.fixture
def quota(tmp_path, make_config):
    """QuotaTracker with test persistence path."""
    config = make_config(["groq", "gemini", "anthropic", "serper"])
    return QuotaTracker(
        provider_configs=config.providers,
        persistence_path=str(tmp_path / "quota.json"),
    )


# ── Registry fixture ─────────────────────────────────────────


@pytest.fixture
def make_registry(make_config):
    """Factory fixture — returns a registry with specified providers."""

    def _factory(providers: list[str]) -> ProviderRegistry:
        config = make_config(providers)
        return ProviderRegistry(config.providers, httpx.AsyncClient())

    return _factory


# ── Mock registry (for planner tests) ────────────────────────


@pytest.fixture
def mock_registry():
    """A MagicMock registry with call_free_llm as AsyncMock."""
    registry = MagicMock()
    registry.call_free_llm = AsyncMock()
    return registry
