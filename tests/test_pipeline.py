"""Tests for smartsplit.pipeline — core pipeline logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from smartsplit.config import SmartSplitConfig
from smartsplit.models import TaskType
from smartsplit.proxy.pipeline import (
    PipelineResult,
    ProxyContext,
    _AnthropicPassthroughError,
    _build_anthropic_text_response,
    _truncate_messages,
    forward_to_brain,
    process_anthropic_request,
    process_anthropic_request_lite,
)

# ── Helpers ──────────────────────────────────────────────────


def _make_ctx(
    brain_name: str = "groq",
    enabled: bool = True,
    detector: object = None,
    pattern_learner: object = None,
) -> ProxyContext:
    """Build a minimal ProxyContext for testing."""
    config = MagicMock(spec=SmartSplitConfig)
    config.providers = {}
    registry = MagicMock()
    registry.brain_name = brain_name
    registry.get = MagicMock(return_value=None)
    registry.get_search_providers = MagicMock(return_value=[])
    registry.get_llm_providers = MagicMock(return_value=[brain_name])
    registry.call_brain = AsyncMock()
    registry.proxy_to_brain = AsyncMock()

    planner = MagicMock()
    router = MagicMock()
    quota = MagicMock()
    bandit = MagicMock()
    http = MagicMock(spec=httpx.AsyncClient)

    return ProxyContext(
        config=config,
        registry=registry,
        planner=planner,
        router=router,
        quota=quota,
        bandit=bandit,
        http=http,
        enabled=enabled,
        detector=detector,
        pattern_learner=pattern_learner,
    )


# ── _truncate_messages ───────────────────────────────────────


class TestTruncateMessages:
    def test_all_fit(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        result = _truncate_messages(msgs, 1000)
        assert len(result) == 2

    def test_drops_old_messages(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a" * 500},
            {"role": "assistant", "content": "b" * 500},
            {"role": "user", "content": "c" * 100},
        ]
        result = _truncate_messages(msgs, 200)
        # system + last user at minimum
        assert any(m["role"] == "system" for m in result)
        assert result[-1]["content"].startswith("c")

    def test_keeps_last_user_when_nothing_fits(self):
        msgs = [
            {"role": "user", "content": "a" * 1000},
        ]
        result = _truncate_messages(msgs, 50)
        assert len(result) >= 1
        assert result[-1]["role"] == "user"

    def test_empty_messages(self):
        result = _truncate_messages([], 1000)
        assert result == []

    def test_system_always_kept(self):
        msgs = [
            {"role": "system", "content": "important"},
            {"role": "user", "content": "x" * 10000},
        ]
        result = _truncate_messages(msgs, 500)
        assert any(m["role"] == "system" for m in result)


# ── PipelineResult ───────────────────────────────────────────


class TestPipelineResult:
    def test_defaults(self):
        r = PipelineResult()
        assert r.status_code == 200
        assert r.body is None
        assert r.is_streaming is False

    def test_with_body(self):
        r = PipelineResult(body={"type": "message"})
        assert r.body["type"] == "message"

    def test_streaming(self):
        r = PipelineResult(streaming_response=MagicMock())
        assert r.is_streaming is True


# ── _AnthropicPassthroughError ───────────────────────────────


class TestAnthropicPassthroughError:
    def test_attributes(self):
        err = _AnthropicPassthroughError(429, {"type": "error"})
        assert err.status_code == 429
        assert err.body == {"type": "error"}
        assert str(err) == "429"


# ── _build_anthropic_text_response ───────────────────────────


class TestBuildAnthropicTextResponse:
    def test_structure(self):
        resp = _build_anthropic_text_response("Hello", "claude-sonnet-4-20250514", 10, 5)
        assert resp["type"] == "message"
        assert resp["role"] == "assistant"
        assert resp["content"][0]["type"] == "text"
        assert resp["content"][0]["text"] == "Hello"
        assert resp["usage"]["input_tokens"] == 10
        assert resp["usage"]["output_tokens"] == 5
        assert resp["stop_reason"] == "end_turn"

    def test_empty_model_defaults(self):
        resp = _build_anthropic_text_response("x", "", 0, 0)
        assert resp["model"] == "smartsplit"

    def test_id_format(self):
        resp = _build_anthropic_text_response("x", "model", 0, 0)
        assert resp["id"].startswith("msg_")


# ── ProxyContext ─────────────────────────────────────────────


class TestProxyContext:
    def test_defaults(self):
        ctx = _make_ctx()
        assert ctx.enabled is True
        assert ctx.enrichment_skip_until == 0.0
        assert ctx.anticipation_stats["requests_with_tools"] == 0

    def test_pending_fakes_empty(self):
        ctx = _make_ctx()
        assert ctx.pending_fakes == {}


# ── forward_to_brain ─────────────────────────────────────────


class TestForwardToBrain:
    @pytest.mark.asyncio
    async def test_success(self):
        ctx = _make_ctx()
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        ctx.registry.call_brain = AsyncMock(return_value=("Response text", usage))

        content, results = await forward_to_brain(ctx, "hello")
        assert content == "Response text"
        assert len(results) == 1
        assert results[0].type == TaskType.GENERAL

    @pytest.mark.asyncio
    async def test_failure_returns_none(self):
        ctx = _make_ctx()
        ctx.registry.call_brain = AsyncMock(side_effect=Exception("fail"))

        content, results = await forward_to_brain(ctx, "hello")
        assert content is None
        assert results == []


# ── process_anthropic_request_lite ───────────────────────────


class TestProcessAnthropicRequestLite:
    @pytest.mark.asyncio
    async def test_passthrough(self):
        ctx = _make_ctx()
        body = {"messages": [{"role": "user", "content": "hi"}]}
        action = await process_anthropic_request_lite(ctx, body, {})
        assert action["type"] == "passthrough"

    @pytest.mark.asyncio
    async def test_with_tools_no_detector(self):
        ctx = _make_ctx(detector=None)
        body = {
            "messages": [{"role": "user", "content": "read proxy.py"}],
            "tools": [{"name": "Read"}],
        }
        action = await process_anthropic_request_lite(ctx, body, {})
        assert action["type"] == "passthrough"

    @pytest.mark.asyncio
    async def test_internal_agent_call_skips_pipeline(self):
        """Agent-internal calls (auto-compact, title gen, …) short-circuit to passthrough
        without firing any SmartSplit LLM call, even if detector and pattern_learner are wired."""
        detector = MagicMock()
        detector.predict = AsyncMock()
        pattern_learner = MagicMock()
        pattern_learner.observe_outcome = MagicMock()

        ctx = _make_ctx(detector=detector, pattern_learner=pattern_learner)
        ctx.registry.call_free_llm = AsyncMock()

        body = {
            "system": "Summarize the conversation below.",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "..."}],
            "tools": [{"name": "Read"}],  # tools may be present even on internal calls
        }
        action = await process_anthropic_request_lite(ctx, body, {})

        assert action["type"] == "passthrough"
        detector.predict.assert_not_awaited()
        pattern_learner.observe_outcome.assert_not_called()
        ctx.registry.call_free_llm.assert_not_awaited()


# ── process_anthropic_request ────────────────────────────────


class TestProcessAnthropicRequest:
    @pytest.mark.asyncio
    async def test_missing_messages(self):
        ctx = _make_ctx()
        result = await process_anthropic_request(ctx, {"model": "x"}, {})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_messages_type(self):
        ctx = _make_ctx()
        result = await process_anthropic_request(ctx, {"messages": "not a list"}, {})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_passthrough_calls_brain(self):
        ctx = _make_ctx(brain_name="groq")
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        ctx.registry.call_brain = AsyncMock(return_value=("OK", usage))

        body = {"messages": [{"role": "user", "content": "hello"}]}
        result = await process_anthropic_request(ctx, body, {})
        assert result.status_code == 200
        assert result.body is not None

    @pytest.mark.asyncio
    async def test_brain_failure_returns_503(self):
        ctx = _make_ctx(brain_name="groq")
        ctx.registry.call_brain = AsyncMock(side_effect=Exception("fail"))

        body = {"messages": [{"role": "user", "content": "hello"}]}
        result = await process_anthropic_request(ctx, body, {})
        assert result.status_code == 503
