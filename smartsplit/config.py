"""Configuration management for SmartSplit.

Supports three config sources (lowest → highest priority):
  1. Built-in defaults
  2. JSON config file  (smartsplit.json / ~/.smartsplit/config.json)
  3. Environment variables  (GROQ_API_KEY, SMARTSPLIT_MODE, …)
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

from smartsplit.models import ContextTier, Mode, ProviderType

# ── Pydantic config models ──────────────────────────────────


class ProviderConfig(BaseModel):
    """Configuration for a single LLM or search provider."""

    api_key: str = ""
    type: ProviderType = ProviderType.FREE
    enabled: bool = False
    limits: dict[str, int] = Field(default_factory=dict)
    daily_budget_tokens: int = 100_000  # reserved for future budget enforcement
    fast_model: str = ""  # cheap model for simple tasks (paid providers only)
    strong_model: str = ""  # best model for complex tasks (paid providers only)
    model: str = ""  # default model for this provider
    temperature: float = 0.3
    max_tokens: int = 4096
    max_search_results: int = 5
    context_tier: ContextTier = ContextTier.SMALL


DEFAULT_FREE_LLM_PRIORITY = ["cerebras", "groq", "gemini", "openrouter", "mistral", "huggingface", "cloudflare"]

# Brain priority: paid first (best quality), then free by capability.
_BRAIN_PRIORITY = ["anthropic", "openai", "deepseek", "groq", "gemini", "openrouter", "mistral", "cerebras"]


class SmartSplitConfig(BaseModel):
    """Top-level SmartSplit configuration."""

    mode: Mode = Mode.BALANCED
    brain: str = Field(default="", description="Main LLM provider. Auto-detected if empty.")
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    competence_table: dict[str, dict[str, int]] = Field(default_factory=dict)
    free_llm_priority: list[str] = Field(default_factory=lambda: list(DEFAULT_FREE_LLM_PRIORITY))
    overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Force a specific provider for a task type. Example: {'code': 'anthropic'}",
    )


# ── Defaults ─────────────────────────────────────────────────

DEFAULT_PROVIDERS: dict[str, dict] = {
    "groq": {
        "type": "free",
        "enabled": True,
        "limits": {"rpm": 1000, "rpd": 14400},
        "model": "llama-3.3-70b-versatile",
        "context_tier": "small",
    },
    "cerebras": {
        "type": "free",
        "enabled": False,
        "limits": {"rpm": 30, "rpd": 14400},
        "model": "qwen-3-235b-a22b-instruct-2507",
        "context_tier": "small",
    },
    "gemini": {
        "type": "free",
        "enabled": True,
        "limits": {"rpm": 15, "rpd": 1500},
        "model": "gemini-2.5-flash",
        "context_tier": "large",
    },
    "deepseek": {
        "type": "paid",
        "enabled": False,
        "limits": {"rpd": 50000},
        "model": "deepseek-chat",
        "fast_model": "deepseek-chat",
        "strong_model": "deepseek-reasoner",
    },
    "openrouter": {
        "type": "free",
        "enabled": False,
        "limits": {"rpm": 20, "rpd": 50},
        "model": "qwen/qwen3-coder:free",
        "context_tier": "medium",
    },
    "mistral": {
        "type": "free",
        "enabled": False,
        "limits": {"rpm": 60, "rpd": 1000000},
        "model": "mistral-small-latest",
        "context_tier": "medium",
    },
    "huggingface": {
        "type": "free",
        "enabled": False,
        "limits": {"rpm": 600},
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "context_tier": "small",
    },
    "cloudflare": {
        "type": "free",
        "enabled": False,
        "limits": {"rpd": 10000},
        "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "context_tier": "small",
    },
    "serper": {
        "type": "free",
        "enabled": True,
        "limits": {"rpm": 100, "monthly": 2500},
    },
    "tavily": {
        "type": "free",
        "enabled": False,
        "limits": {"monthly": 1000},
    },
    "anthropic": {
        "type": "paid",
        "enabled": False,
        "daily_budget_tokens": 100_000,
        "model": "claude-sonnet-4-6-20250514",
        "fast_model": "claude-haiku-4-5-20251001",
        "strong_model": "claude-sonnet-4-6-20250514",
        "context_tier": "large",
    },
    "openai": {
        "type": "paid",
        "enabled": False,
        "daily_budget_tokens": 100_000,
        "model": "gpt-4o",
        "fast_model": "gpt-4o-mini",
        "strong_model": "o3",
        "context_tier": "large",
    },
}

DEFAULT_COMPETENCE_TABLE: dict[str, dict[str, int]] = {
    "web_search": {
        "serper": 9,
        "tavily": 9,
        "groq": 3,
        "cerebras": 3,
        "gemini": 7,
        "deepseek:fast": 3,
        "deepseek:strong": 3,
        "openrouter": 3,
        "huggingface": 3,
        "cloudflare": 3,
        "anthropic:fast": 5,
        "anthropic:strong": 5,
        "openai:fast": 5,
        "openai:strong": 5,
    },
    "summarize": {
        "groq": 8,
        "cerebras": 9,
        "gemini": 8,
        "deepseek:fast": 8,
        "deepseek:strong": 9,
        "openrouter": 8,
        "mistral": 8,
        "huggingface": 6,
        "cloudflare": 6,
        "anthropic:fast": 8,
        "anthropic:strong": 9,
        "openai:fast": 7,
        "openai:strong": 8,
    },
    # Scores for free providers use the single available model.
    # Paid providers have two tiers:
    #   "provider:fast"   — cheap model (haiku, gpt-4o-mini) for simple tasks
    #   "provider:strong" — best model (sonnet, gpt-4o) for complex tasks
    # The router picks the tier based on subtask complexity.
    "code": {
        "groq": 7,
        "cerebras": 8,
        "gemini": 8,
        "deepseek:fast": 9,
        "deepseek:strong": 10,
        "openrouter": 10,
        "mistral": 6,
        "huggingface": 7,
        "cloudflare": 5,
        "anthropic:fast": 7,
        "anthropic:strong": 9,
        "openai:fast": 7,
        "openai:strong": 9,
    },
    "reasoning": {
        "groq": 6,
        "cerebras": 9,
        "gemini": 8,
        "deepseek:fast": 8,
        "deepseek:strong": 9,
        "openrouter": 8,
        "mistral": 5,
        "huggingface": 5,
        "cloudflare": 4,
        "anthropic:fast": 6,
        "anthropic:strong": 9,
        "openai:fast": 6,
        "openai:strong": 9,
    },
    "translation": {
        "groq": 7,
        "cerebras": 7,
        "gemini": 8,
        "deepseek:fast": 7,
        "deepseek:strong": 8,
        "openrouter": 7,
        "mistral": 9,
        "huggingface": 5,
        "cloudflare": 5,
        "anthropic:fast": 7,
        "anthropic:strong": 8,
        "openai:fast": 7,
        "openai:strong": 8,
    },
    "boilerplate": {
        "groq": 9,
        "cerebras": 9,
        "gemini": 8,
        "deepseek:fast": 8,
        "deepseek:strong": 9,
        "openrouter": 8,
        "mistral": 8,
        "huggingface": 6,
        "cloudflare": 6,
        "anthropic:fast": 8,
        "anthropic:strong": 7,
        "openai:fast": 8,
        "openai:strong": 7,
    },
    "general": {
        "groq": 7,
        "cerebras": 9,
        "gemini": 8,
        "deepseek:fast": 8,
        "deepseek:strong": 9,
        "openrouter": 8,
        "mistral": 7,
        "huggingface": 5,
        "cloudflare": 5,
        "anthropic:fast": 7,
        "anthropic:strong": 9,
        "openai:fast": 7,
        "openai:strong": 8,
    },
    "math": {
        "groq": 5,
        "cerebras": 7,
        "gemini": 8,
        "deepseek:fast": 9,
        "deepseek:strong": 10,
        "openrouter": 8,
        "mistral": 5,
        "huggingface": 4,
        "cloudflare": 3,
        "anthropic:fast": 5,
        "anthropic:strong": 9,
        "openai:fast": 5,
        "openai:strong": 9,
    },
    "creative": {
        "groq": 5,
        "cerebras": 6,
        "gemini": 7,
        "deepseek:fast": 6,
        "deepseek:strong": 7,
        "openrouter": 7,
        "mistral": 6,
        "huggingface": 4,
        "cloudflare": 4,
        "anthropic:fast": 6,
        "anthropic:strong": 9,
        "openai:fast": 6,
        "openai:strong": 8,
    },
    "factual": {
        "groq": 8,
        "cerebras": 9,
        "gemini": 9,
        "deepseek:fast": 7,
        "deepseek:strong": 8,
        "openrouter": 7,
        "mistral": 7,
        "huggingface": 5,
        "cloudflare": 5,
        "anthropic:fast": 7,
        "anthropic:strong": 9,
        "openai:fast": 7,
        "openai:strong": 8,
    },
    "extraction": {
        "groq": 8,
        "cerebras": 9,
        "gemini": 8,
        "deepseek:fast": 8,
        "deepseek:strong": 9,
        "openrouter": 7,
        "mistral": 7,
        "huggingface": 5,
        "cloudflare": 5,
        "anthropic:fast": 8,
        "anthropic:strong": 9,
        "openai:fast": 7,
        "openai:strong": 8,
    },
}

# ── Env-var mapping ──────────────────────────────────────────

_ENV_KEY_MAP: dict[str, str] = {
    "GROQ_API_KEY": "groq",
    "CEREBRAS_API_KEY": "cerebras",
    "GEMINI_API_KEY": "gemini",
    "DEEPSEEK_API_KEY": "deepseek",
    "OPENROUTER_API_KEY": "openrouter",
    "MISTRAL_API_KEY": "mistral",
    "HF_TOKEN": "huggingface",
    "CLOUDFLARE_API_KEY": "cloudflare",
    "SERPER_API_KEY": "serper",
    "TAVILY_API_KEY": "tavily",
    "ANTHROPIC_API_KEY": "anthropic",
    "OPENAI_API_KEY": "openai",
}

# ── Loader ───────────────────────────────────────────────────


def find_config_path() -> Path:
    """Find config file in standard locations."""
    candidates = [
        Path.cwd() / "smartsplit.json",
        Path.home() / ".smartsplit" / "config.json",
        Path.home() / ".config" / "smartsplit" / "config.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_config() -> SmartSplitConfig:
    """Load, merge, validate, and return the config as a Pydantic model."""
    raw: dict = {
        "mode": "balanced",
        "providers": copy.deepcopy(DEFAULT_PROVIDERS),
        "competence_table": copy.deepcopy(DEFAULT_COMPETENCE_TABLE),
    }

    # Layer 2: JSON file
    config_path = find_config_path()
    if config_path.exists():
        with open(config_path) as f:
            _deep_merge(raw, json.load(f))

    # Layer 3: environment variables
    mode_env = os.environ.get("SMARTSPLIT_MODE")
    if mode_env:
        raw["mode"] = mode_env

    brain_env = os.environ.get("SMARTSPLIT_BRAIN")
    if brain_env:
        raw["brain"] = brain_env

    for env_var, provider_name in _ENV_KEY_MAP.items():
        api_key = os.environ.get(env_var)
        if api_key:
            raw.setdefault("providers", {}).setdefault(provider_name, {})
            raw["providers"][provider_name]["api_key"] = api_key
            raw["providers"][provider_name]["enabled"] = True

    cfg = SmartSplitConfig.model_validate(raw)

    # Auto-detect brain if not explicitly set
    if not cfg.brain:
        cfg = cfg.model_copy(update={"brain": _resolve_brain(cfg.providers)})

    return cfg


def _resolve_brain(providers: dict[str, ProviderConfig]) -> str:
    """Pick the best available provider as the brain (main LLM).

    Priority: paid providers first (best quality), then free by capability.
    Search providers (serper, tavily) are never selected as brain.
    """
    _SEARCH_PROVIDERS = {"serper", "tavily"}
    for name in _BRAIN_PRIORITY:
        pconfig = providers.get(name)
        if pconfig and pconfig.enabled and pconfig.api_key and name not in _SEARCH_PROVIDERS:
            return name
    # Fallback: first enabled LLM provider in any order
    for name, pconfig in providers.items():
        if pconfig.enabled and pconfig.api_key and name not in _SEARCH_PROVIDERS:
            return name
    return ""


# ── Helpers ──────────────────────────────────────────────────


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
