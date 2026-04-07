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

from smartsplit.models import Mode, ProviderType

# ── Pydantic config models ──────────────────────────────────


class ProviderConfig(BaseModel):
    api_key: str = ""
    type: ProviderType = ProviderType.FREE
    enabled: bool = False
    limits: dict[str, int] = Field(default_factory=dict)
    daily_budget_tokens: int = 100_000


class SmartSplitConfig(BaseModel):
    mode: Mode = Mode.BALANCED
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    competence_table: dict[str, dict[str, int]] = Field(default_factory=dict)


# ── Defaults ─────────────────────────────────────────────────

DEFAULT_PROVIDERS: dict[str, dict] = {
    "groq": {"type": "free", "enabled": True, "limits": {"rpm": 30, "rpd": 14400}},
    "cerebras": {"type": "free", "enabled": False, "limits": {"rpm": 30, "rpd": 14400}},
    "gemini": {"type": "free", "enabled": True, "limits": {"rpm": 10, "rpd": 500}},
    "deepseek": {"type": "free", "enabled": False, "limits": {"rpd": 50000}},
    "openrouter": {"type": "free", "enabled": False, "limits": {"rpm": 20, "rpd": 50}},
    "mistral": {"type": "free", "enabled": False, "limits": {"rpm": 10}},
    "serper": {"type": "free", "enabled": True, "limits": {"rpm": 100, "monthly": 2500}},
    "tavily": {"type": "free", "enabled": False, "limits": {"monthly": 1000}},
    "anthropic": {"type": "paid", "enabled": False, "daily_budget_tokens": 100_000},
    "openai": {"type": "paid", "enabled": False, "daily_budget_tokens": 100_000},
}

DEFAULT_COMPETENCE_TABLE: dict[str, dict[str, int]] = {
    "web_search": {
        "serper": 9,
        "tavily": 9,
        "groq": 3,
        "cerebras": 3,
        "gemini": 7,
        "deepseek": 3,
        "openrouter": 3,
        "anthropic": 5,
        "openai": 5,
    },
    "summarize": {
        "groq": 8,
        "cerebras": 9,
        "gemini": 8,
        "deepseek": 8,
        "openrouter": 8,
        "mistral": 7,
        "anthropic": 9,
        "openai": 8,
    },
    "code": {
        "groq": 7,
        "cerebras": 8,
        "gemini": 8,
        "deepseek": 9,
        "openrouter": 10,
        "mistral": 6,
        "anthropic": 10,
        "openai": 7,
    },
    "reasoning": {
        "groq": 6,
        "cerebras": 9,
        "gemini": 8,
        "deepseek": 8,
        "openrouter": 8,
        "mistral": 5,
        "anthropic": 10,
        "openai": 7,
    },
    "translation": {
        "groq": 7,
        "cerebras": 7,
        "gemini": 8,
        "deepseek": 7,
        "openrouter": 7,
        "mistral": 9,
        "anthropic": 8,
        "openai": 8,
    },
    "boilerplate": {
        "groq": 9,
        "cerebras": 9,
        "gemini": 8,
        "deepseek": 8,
        "openrouter": 8,
        "mistral": 7,
        "anthropic": 7,
        "openai": 7,
    },
    "general": {
        "groq": 7,
        "cerebras": 9,
        "gemini": 8,
        "deepseek": 8,
        "openrouter": 8,
        "mistral": 6,
        "anthropic": 9,
        "openai": 8,
    },
    "math": {
        "groq": 5,
        "cerebras": 7,
        "gemini": 8,
        "deepseek": 9,
        "openrouter": 8,
        "mistral": 5,
        "anthropic": 10,
        "openai": 7,
    },
    "creative": {
        "groq": 5,
        "cerebras": 6,
        "gemini": 7,
        "deepseek": 6,
        "openrouter": 7,
        "mistral": 6,
        "anthropic": 10,
        "openai": 8,
    },
    "factual": {
        "groq": 8,
        "cerebras": 9,
        "gemini": 9,
        "deepseek": 7,
        "openrouter": 7,
        "mistral": 6,
        "anthropic": 8,
        "openai": 8,
    },
    "extraction": {
        "groq": 8,
        "cerebras": 9,
        "gemini": 8,
        "deepseek": 8,
        "openrouter": 7,
        "mistral": 7,
        "anthropic": 9,
        "openai": 8,
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

    for env_var, provider_name in _ENV_KEY_MAP.items():
        api_key = os.environ.get(env_var)
        if api_key:
            raw.setdefault("providers", {}).setdefault(provider_name, {})
            raw["providers"][provider_name]["api_key"] = api_key
            raw["providers"][provider_name]["enabled"] = True

    return SmartSplitConfig.model_validate(raw)


# ── Helpers ──────────────────────────────────────────────────


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
