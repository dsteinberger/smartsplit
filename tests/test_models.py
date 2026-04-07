"""Tests for SmartSplit domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from smartsplit.models import (
    Complexity,
    Mode,
    ProviderStatus,
    ProviderType,
    RouteResult,
    SavingsReport,
    Subtask,
    TaskType,
)

# ── Enums ────────────────────────────────────────────────────


class TestEnums:
    @pytest.mark.parametrize(
        "enum_cls, value",
        [
            pytest.param(TaskType, "web_search", id="task-web_search"),
            pytest.param(TaskType, "code", id="task-code"),
            pytest.param(TaskType, "general", id="task-general"),
            pytest.param(TaskType, "math", id="task-math"),
            pytest.param(TaskType, "creative", id="task-creative"),
            pytest.param(TaskType, "factual", id="task-factual"),
            pytest.param(TaskType, "extraction", id="task-extraction"),
            pytest.param(Mode, "economy", id="mode-economy"),
            pytest.param(Mode, "balanced", id="mode-balanced"),
            pytest.param(Mode, "quality", id="mode-quality"),
            pytest.param(Complexity, "low", id="complexity-low"),
            pytest.param(Complexity, "high", id="complexity-high"),
            pytest.param(ProviderType, "free", id="provider-free"),
            pytest.param(ProviderType, "paid", id="provider-paid"),
        ],
    )
    def test_valid_enum_values(self, enum_cls, value):
        assert enum_cls(value) == value

    @pytest.mark.parametrize(
        "enum_cls, bad_value",
        [
            pytest.param(TaskType, "invalid_type", id="bad-task-type"),
            pytest.param(Mode, "turbo", id="bad-mode"),
            pytest.param(Complexity, "extreme", id="bad-complexity"),
            pytest.param(ProviderType, "premium", id="bad-provider-type"),
        ],
    )
    def test_invalid_enum_values_raise(self, enum_cls, bad_value):
        with pytest.raises(ValueError):
            enum_cls(bad_value)


# ── Subtask ──────────────────────────────────────────────────


class TestSubtask:
    def test_defaults(self):
        s = Subtask(content="do something")
        assert s.type == TaskType.GENERAL
        assert s.complexity == Complexity.MEDIUM

    @pytest.mark.parametrize(
        "data, expected_type, expected_complexity",
        [
            pytest.param(
                {"type": "code", "content": "hello", "complexity": "high"},
                TaskType.CODE,
                Complexity.HIGH,
                id="code-high",
            ),
            pytest.param(
                {"type": "web_search", "content": "query", "complexity": "low"},
                TaskType.WEB_SEARCH,
                Complexity.LOW,
                id="search-low",
            ),
            pytest.param(
                {"content": "minimal"},
                TaskType.GENERAL,
                Complexity.MEDIUM,
                id="defaults-only",
            ),
        ],
    )
    def test_from_dict(self, data, expected_type, expected_complexity):
        s = Subtask.model_validate(data)
        assert s.type == expected_type
        assert s.complexity == expected_complexity

    @pytest.mark.parametrize(
        "bad_data",
        [
            pytest.param({"type": "invalid_type", "content": "x"}, id="bad-type"),
            pytest.param({"type": "code"}, id="missing-content"),
            pytest.param({"content": "x", "complexity": "extreme"}, id="bad-complexity"),
        ],
    )
    def test_invalid_subtask_rejected(self, bad_data):
        with pytest.raises(ValidationError):
            Subtask.model_validate(bad_data)

    def test_empty_content_accepted(self):
        s = Subtask(content="")
        assert s.content == ""


# ── RouteResult ──────────────────────────────────────────────


class TestRouteResult:
    def test_creation(self):
        r = RouteResult(type=TaskType.CODE, response="print('hi')", provider="groq", score=0.85)
        assert r.provider == "groq"
        assert r.score == 0.85

    def test_default_score(self):
        r = RouteResult(type=TaskType.GENERAL, response="ok", provider="none")
        assert r.score == 0.0

    def test_negative_score_allowed(self):
        r = RouteResult(type=TaskType.GENERAL, response="ok", provider="x", score=-1.0)
        assert r.score == -1.0


# ── SavingsReport ────────────────────────────────────────────


class TestSavingsReport:
    def test_creation(self):
        r = SavingsReport(
            total_requests=10,
            free_requests=8,
            paid_requests=2,
            free_percentage=80.0,
            estimated_tokens_saved=12000,
            estimated_cost_saved_usd=0.036,
        )
        assert r.free_percentage == 80.0

    def test_zero_requests(self):
        r = SavingsReport(
            total_requests=0,
            free_requests=0,
            paid_requests=0,
            free_percentage=0.0,
            estimated_tokens_saved=0,
            estimated_cost_saved_usd=0.0,
        )
        assert r.total_requests == 0


# ── ProviderStatus ───────────────────────────────────────────


class TestProviderStatus:
    @pytest.mark.parametrize(
        "availability",
        [
            pytest.param(0.0, id="zero"),
            pytest.param(0.5, id="half"),
            pytest.param(1.0, id="full"),
        ],
    )
    def test_valid_availability(self, availability):
        ps = ProviderStatus(name="test", type=ProviderType.FREE, enabled=True, availability=availability)
        assert ps.availability == availability

    @pytest.mark.parametrize(
        "bad_value",
        [
            pytest.param(-0.1, id="negative"),
            pytest.param(1.5, id="above-one"),
        ],
    )
    def test_availability_bounds_rejected(self, bad_value):
        with pytest.raises(ValidationError):
            ProviderStatus(name="test", type=ProviderType.FREE, enabled=True, availability=bad_value)
