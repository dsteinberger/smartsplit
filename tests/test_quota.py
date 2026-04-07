"""Tests for SmartSplit quota tracker and token estimation."""

from __future__ import annotations

import time

import pytest

from smartsplit.models import SavingsReport
from smartsplit.quota import QuotaTracker, estimate_tokens

# ── Token Estimation ────────────────────────────────────────


class TestTokenEstimation:
    def test_empty_prompt_returns_fallback(self):
        assert estimate_tokens("") == 500

    def test_short_prompt_returns_minimum(self):
        assert estimate_tokens("hello") >= 500

    def test_long_prompt_scales_with_length(self):
        prompt = "x" * 4000  # ~1000 tokens * 1.2 = 1200
        tokens = estimate_tokens(prompt)
        assert tokens == 1200

    def test_safety_margin_applied(self):
        prompt = "x" * 2000  # 500 raw tokens → 500 * 1.2 = 600
        tokens = estimate_tokens(prompt)
        assert tokens == 600

    def test_realistic_prompt(self):
        prompt = "Write a Python function that calculates the Fibonacci sequence using memoization"
        tokens = estimate_tokens(prompt)
        # ~80 chars / 4 * 1.2 = ~24 → but floor is 500
        assert tokens >= 500


# ── Recording ────────────────────────────────────────────────


class TestRecording:
    def test_free_usage(self, quota):
        prompt = "Summarize this text about machine learning and natural language processing"
        quota.record_usage("groq", "summarize", prompt=prompt)
        assert quota.get_usage("groq") == 1
        report = quota.get_savings_report()
        assert report.free_requests == 1
        assert report.paid_requests == 0
        assert report.estimated_tokens_saved == estimate_tokens(prompt)

    def test_free_usage_without_prompt(self, quota):
        """Without a prompt, uses the fallback token estimate."""
        quota.record_usage("groq", "summarize")
        report = quota.get_savings_report()
        assert report.estimated_tokens_saved == 500  # fallback

    def test_paid_usage(self, quota):
        quota.record_usage("anthropic", "code", is_paid=True)
        report = quota.get_savings_report()
        assert report.paid_requests == 1
        assert report.estimated_tokens_saved == 0

    @pytest.mark.parametrize(
        "calls, expected_count",
        [
            pytest.param(1, 1, id="single"),
            pytest.param(5, 5, id="five"),
            pytest.param(100, 100, id="hundred"),
        ],
    )
    def test_multiple_calls(self, quota, calls, expected_count):
        for _ in range(calls):
            quota.record_usage("groq", "summarize")
        assert quota.get_usage("groq") == expected_count

    def test_tracks_by_type(self, quota):
        quota.record_usage("groq", "summarize")
        quota.record_usage("groq", "summarize")
        quota.record_usage("groq", "code")
        entry = quota._usage["groq"]
        assert entry["by_type"]["summarize"] == 2
        assert entry["by_type"]["code"] == 1

    def test_unknown_provider(self, quota):
        quota.record_usage("brand_new_provider", "general")
        assert quota.get_usage("brand_new_provider") == 1


# ── Availability ─────────────────────────────────────────────


class TestAvailability:
    def test_fresh_is_full(self, quota):
        assert quota.get_availability("groq") == 1.0

    @pytest.mark.parametrize(
        "provider, used, limit, expected",
        [
            pytest.param("groq", 7200, 14400, 0.5, id="groq-half"),
            pytest.param("groq", 14400, 14400, 0.0, id="groq-at-limit"),
            pytest.param("groq", 0, 14400, 1.0, id="groq-fresh"),
            pytest.param("gemini", 1500, 1500, 0.0, id="gemini-at-limit"),
            pytest.param("gemini", 750, 1500, 0.5, id="gemini-half"),
        ],
    )
    def test_availability_ratios(self, quota, provider, used, limit, expected):
        quota._usage[provider] = {"count": used, "last_reset": time.time(), "by_type": {}}
        assert abs(quota.get_availability(provider) - expected) < 0.01

    def test_daily_reset(self, quota):
        quota._usage["groq"] = {"count": 10000, "last_reset": time.time() - 200000, "by_type": {}}
        assert quota.get_availability("groq") == 1.0

    def test_unknown_provider_uses_default(self, quota):
        quota._usage["unknown"] = {"count": 500, "last_reset": time.time(), "by_type": {}}
        assert abs(quota.get_availability("unknown") - 0.5) < 0.01


# ── Savings report ───────────────────────────────────────────


class TestSavingsReport:
    def test_report_types(self, quota):
        quota.record_usage("groq", "summarize")
        report = quota.get_savings_report()
        assert isinstance(report, SavingsReport)

    def test_mixed_usage(self, quota):
        p1 = "Summarize the key findings of this research paper"
        p2 = "Write a function to sort a list"
        quota.record_usage("groq", "summarize", prompt=p1)
        quota.record_usage("groq", "code", prompt=p2)
        quota.record_usage("anthropic", "reasoning", is_paid=True, prompt="analyze this")
        report = quota.get_savings_report()
        assert report.total_requests == 3
        assert report.free_requests == 2
        assert report.paid_requests == 1
        assert report.free_percentage > 60
        assert report.estimated_tokens_saved == estimate_tokens(p1) + estimate_tokens(p2)

    def test_zero_requests(self, quota):
        report = quota.get_savings_report()
        assert report.total_requests == 0
        assert report.free_percentage == 0.0

    def test_providers_usage_dict(self, quota):
        quota.record_usage("groq", "code")
        quota.record_usage("serper", "web_search")
        report = quota.get_savings_report()
        assert report.providers_usage["groq"] == 1
        assert report.providers_usage["serper"] == 1


# ── Persistence ──────────────────────────────────────────────


class TestPersistence:
    def test_save_and_load(self, tmp_path, make_config):
        config = make_config(["groq", "gemini"])
        qt1 = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        qt1.record_usage("groq", "summarize")
        qt1.record_usage("groq", "code")
        qt1.flush()

        qt2 = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "q.json"))
        assert qt2.get_usage("groq") == 2
        assert qt2.get_savings_report().free_requests == 2

    def test_missing_file_starts_fresh(self, tmp_path, make_config):
        config = make_config(["groq"])
        qt = QuotaTracker(provider_configs=config.providers, persistence_path=str(tmp_path / "nope.json"))
        assert qt.get_usage("groq") == 0

    def test_corrupted_file_starts_fresh(self, tmp_path, make_config):
        config = make_config(["groq"])
        bad_file = tmp_path / "quota.json"
        bad_file.write_text("NOT JSON {{{")
        qt = QuotaTracker(provider_configs=config.providers, persistence_path=str(bad_file))
        assert qt.get_usage("groq") == 0
