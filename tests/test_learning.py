"""Tests for MAB (Multi-Armed Bandit) adaptive scoring."""

from __future__ import annotations

from smartsplit.routing.learning import BanditScorer


class TestBanditScorer:
    def test_no_data_returns_prior(self, tmp_path):
        bandit = BanditScorer(persistence_path=str(tmp_path / "scores.json"))
        # No data → returns the prior
        assert bandit.score("code", "groq", prior=0.7) == 0.7

    def test_few_data_blends_with_prior(self, tmp_path):
        bandit = BanditScorer(persistence_path=str(tmp_path / "scores.json"))
        bandit.record("code", "groq", success=True)
        bandit.record("code", "groq", success=True)
        # With 2 data points, prior still dominates (prior_strength=10)
        score = bandit.score("code", "groq", prior=0.7)
        # Score should be close to prior, not wildly different
        assert 0.5 < score < 2.0

    def test_enough_data_uses_learned_score(self, tmp_path):
        bandit = BanditScorer(persistence_path=str(tmp_path / "scores.json"))
        for _ in range(10):
            bandit.record("code", "groq", success=True)
        # 10 successes → score should be > prior of 0.5
        score = bandit.score("code", "groq", prior=0.5)
        assert score > 0.5

    def test_failures_lower_score(self, tmp_path):
        bandit = BanditScorer(persistence_path=str(tmp_path / "scores.json"))
        # Need enough data so exploration bonus is small
        for _ in range(50):
            bandit.record("code", "groq", success=True)
        for _ in range(50):
            bandit.record("code", "groq", success=False)
        # 50% success rate with enough data → score should be around 0.5-0.7
        score = bandit.score("code", "groq", prior=0.9)
        assert score < 0.9  # learned score is lower than inflated prior

    def test_good_provider_beats_bad_provider(self, tmp_path):
        bandit = BanditScorer(persistence_path=str(tmp_path / "scores.json"))
        for _ in range(20):
            bandit.record("code", "good_provider", success=True)
            bandit.record("code", "bad_provider", success=False)
        good = bandit.score("code", "good_provider", prior=0.5)
        bad = bandit.score("code", "bad_provider", prior=0.5)
        assert good > bad

    def test_independent_task_types(self, tmp_path):
        bandit = BanditScorer(persistence_path=str(tmp_path / "scores.json"))
        for _ in range(10):
            bandit.record("code", "groq", success=True)
            bandit.record("translation", "groq", success=False)
        code_score = bandit.score("code", "groq", prior=0.5)
        trans_score = bandit.score("translation", "groq", prior=0.5)
        assert code_score > trans_score

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "scores.json")
        bandit1 = BanditScorer(persistence_path=path)
        for _ in range(5):
            bandit1.record("code", "groq", success=True)
        bandit1.flush()

        bandit2 = BanditScorer(persistence_path=path)
        # Should load the saved data
        assert bandit2.score("code", "groq", prior=0.5) > 0.5

    def test_get_stats(self, tmp_path):
        bandit = BanditScorer(persistence_path=str(tmp_path / "scores.json"))
        bandit.record("code", "groq", success=True)
        bandit.record("code", "groq", success=False)
        stats = bandit.get_stats()
        assert stats["code"]["groq"]["success"] == 1
        assert stats["code"]["groq"]["total"] == 2
