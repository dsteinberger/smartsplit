"""Tests for SmartSplit config module."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from smartsplit.config import ProviderConfig, SmartSplitConfig, _deep_merge, load_config
from smartsplit.models import Mode, ProviderType

# ── _deep_merge ──────────────────────────────────────────────


class TestDeepMerge:
    @pytest.mark.parametrize(
        "base, override, expected",
        [
            pytest.param(
                {"a": 1, "b": 2},
                {"b": 3, "c": 4},
                {"a": 1, "b": 3, "c": 4},
                id="flat-merge",
            ),
            pytest.param(
                {"x": {"y": 1, "z": 2}},
                {"x": {"z": 3}},
                {"x": {"y": 1, "z": 3}},
                id="nested-merge",
            ),
            pytest.param(
                {"a": 1},
                {},
                {"a": 1},
                id="empty-override",
            ),
            pytest.param(
                {},
                {"a": 1},
                {"a": 1},
                id="empty-base",
            ),
            pytest.param(
                {"a": {"b": 1}},
                {"a": "replaced"},
                {"a": "replaced"},
                id="dict-replaced-by-scalar",
            ),
        ],
    )
    def test_merge(self, base, override, expected):
        _deep_merge(base, override)
        assert base == expected


# ── load_config ──────────────────────────────────────────────


def _no_file(tmp_path):
    """Patch helper — config file doesn't exist."""
    return patch("smartsplit.config.find_config_path", return_value=tmp_path / "nope.json")


class TestLoadConfig:
    def test_returns_pydantic_model(self, tmp_path):
        with _no_file(tmp_path), patch.dict(os.environ, {}, clear=True):
            config = load_config()
        assert isinstance(config, SmartSplitConfig)
        assert config.mode == Mode.BALANCED

    def test_defaults_have_all_providers(self, tmp_path):
        with _no_file(tmp_path), patch.dict(os.environ, {}, clear=True):
            config = load_config()
        for name in (
            "groq",
            "cerebras",
            "gemini",
            "deepseek",
            "openrouter",
            "mistral",
            "serper",
            "tavily",
            "anthropic",
            "openai",
        ):
            assert name in config.providers, f"Missing default provider: {name}"

    def test_competence_table_has_all_task_types(self, tmp_path):
        with _no_file(tmp_path), patch.dict(os.environ, {}, clear=True):
            config = load_config()
        for task in (
            "web_search",
            "code",
            "summarize",
            "reasoning",
            "translation",
            "boilerplate",
            "general",
            "math",
            "creative",
            "factual",
            "extraction",
        ):
            assert task in config.competence_table, f"Missing task type: {task}"

    def test_load_from_file(self, tmp_path):
        config_file = tmp_path / "smartsplit.json"
        config_file.write_text(
            json.dumps(
                {
                    "mode": "economy",
                    "providers": {"groq": {"api_key": "gsk_test123"}},
                }
            )
        )
        with patch("smartsplit.config.find_config_path", return_value=config_file):
            with patch.dict(os.environ, {}, clear=True):
                config = load_config()
        assert config.mode == Mode.ECONOMY
        assert config.providers["groq"].api_key == "gsk_test123"
        assert "gemini" in config.providers  # defaults preserved

    @pytest.mark.parametrize(
        "env_var, provider, expected_key",
        [
            pytest.param("GROQ_API_KEY", "groq", "gsk_env", id="groq"),
            pytest.param("ANTHROPIC_API_KEY", "anthropic", "sk-ant-env", id="anthropic"),
            pytest.param("DEEPSEEK_API_KEY", "deepseek", "ds_env", id="deepseek"),
            pytest.param("GEMINI_API_KEY", "gemini", "AIza_env", id="gemini"),
        ],
    )
    def test_env_vars_override(self, tmp_path, env_var, provider, expected_key):
        with _no_file(tmp_path), patch.dict(os.environ, {env_var: expected_key}, clear=True):
            config = load_config()
        assert config.providers[provider].api_key == expected_key
        assert config.providers[provider].enabled is True

    def test_env_mode_override(self, tmp_path):
        with _no_file(tmp_path), patch.dict(os.environ, {"SMARTSPLIT_MODE": "quality"}, clear=True):
            config = load_config()
        assert config.mode == Mode.QUALITY

    def test_deepcopy_isolation(self, tmp_path):
        """Successive loads must not share mutable state."""
        with _no_file(tmp_path), patch.dict(os.environ, {}, clear=True):
            c1 = load_config()
            c2 = load_config()
        c1.providers["groq"].api_key = "mutated"
        assert c2.providers["groq"].api_key != "mutated"

    def test_invalid_json_file_raises(self, tmp_path):
        config_file = tmp_path / "smartsplit.json"
        config_file.write_text("NOT JSON {{{")
        with patch("smartsplit.config.find_config_path", return_value=config_file):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(json.JSONDecodeError):
                    load_config()


# ── ProviderConfig model ─────────────────────────────────────


class TestProviderConfig:
    def test_defaults(self):
        pc = ProviderConfig()
        assert pc.api_key == ""
        assert pc.type == ProviderType.FREE
        assert pc.enabled is False
        assert pc.daily_budget_tokens == 100_000

    @pytest.mark.parametrize(
        "ptype",
        [
            pytest.param("free", id="free"),
            pytest.param("paid", id="paid"),
        ],
    )
    def test_valid_types(self, ptype):
        pc = ProviderConfig(type=ptype)
        assert pc.type == ptype

    def test_invalid_type_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProviderConfig(type="premium")
