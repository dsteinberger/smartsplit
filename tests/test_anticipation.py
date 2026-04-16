"""Tests for SmartSplit anticipation module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smartsplit.tools.anticipation import (
    _fill_missing_args,
    anticipate_anthropic,
    anticipate_tools,
)
from smartsplit.tools.anticipator import ToolResult
from smartsplit.tools.intention_detector import AnticipatedTool, Prediction

# ── Helpers ────────────────────────────────────────────────────


def _make_ctx(
    detector: object | None = "auto",
    anticipator: object | None = "auto",
    pattern_learner: object | None = None,
    registry: object | None = None,
) -> MagicMock:
    """Build a mock ProxyContext with the right attributes."""
    ctx = MagicMock()

    if detector == "auto":
        ctx.detector = AsyncMock()
        ctx.detector.predict = AsyncMock(return_value=Prediction(should_anticipate=False, confidence=0.0))
    else:
        ctx.detector = detector

    if anticipator == "auto":
        ctx.anticipator = AsyncMock()
        ctx.anticipator.execute = AsyncMock(return_value=[])
    else:
        ctx.anticipator = anticipator

    ctx.pattern_learner = pattern_learner
    ctx.registry = registry or MagicMock()
    ctx.registry.call_free_llm = AsyncMock(return_value='["query1"]')

    ctx.anticipation_stats = {
        "requests_with_tools": 0,
        "predictions_made": 0,
        "predictions_skipped": 0,
        "tools_anticipated": 0,
        "tools_injected": 0,
        "tools_failed": 0,
        "files_from_regex": 0,
        "files_already_read": 0,
        "files_recently_written": 0,
    }
    return ctx


def _prediction(tools: list[AnticipatedTool], confidence: float = 0.9) -> Prediction:
    return Prediction(should_anticipate=True, confidence=confidence, tools=tools)


# ── _fill_missing_args ─────────────────────────────────────────


class TestFillMissingArgs:
    @pytest.mark.asyncio
    async def test_read_tool_extracts_paths_from_prompt(self):
        """Read tool without path extracts file paths from user prompt."""
        ctx = _make_ctx()
        predictions = [
            AnticipatedTool(tool="read_file", args={}, reason="read", confidence=0.9),
        ]
        result = await _fill_missing_args(ctx, "req-1", predictions, "please fix auth.py", [])
        assert len(result) == 1
        assert result[0].tool == "read_file"
        assert result[0].args["path"] == "auth.py"

    @pytest.mark.asyncio
    async def test_read_tool_creates_multiple_predictions_for_multiple_files(self):
        """Read tool without path creates one prediction per mentioned file."""
        ctx = _make_ctx()
        predictions = [
            AnticipatedTool(tool="read_file", args={}, reason="read", confidence=0.9),
        ]
        result = await _fill_missing_args(ctx, "req-1", predictions, "compare auth.py and models.py", [])
        assert len(result) == 2
        paths = [r.args["path"] for r in result]
        assert "auth.py" in paths
        assert "models.py" in paths

    @pytest.mark.asyncio
    async def test_read_tool_no_files_in_prompt_skips(self):
        """Read tool without path and no files in prompt is skipped."""
        ctx = _make_ctx()
        predictions = [
            AnticipatedTool(tool="read_file", args={}, reason="read", confidence=0.9),
        ]
        result = await _fill_missing_args(ctx, "req-1", predictions, "explain how auth works", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_list_dir_defaults_to_dot(self):
        """List directory tool without path defaults to '.'."""
        ctx = _make_ctx()
        predictions = [
            AnticipatedTool(tool="list_directory", args={}, reason="list", confidence=0.9),
        ]
        result = await _fill_missing_args(ctx, "req-1", predictions, "show me the project files", [])
        assert len(result) == 1
        assert result[0].tool == "list_directory"
        assert result[0].args["path"] == "."

    @pytest.mark.asyncio
    async def test_search_tool_extracts_query_via_llm(self):
        """Search tool without query calls LLM to extract search queries."""
        ctx = _make_ctx()
        ctx.registry.call_free_llm = AsyncMock(return_value='["best python frameworks 2025"]')
        predictions = [
            AnticipatedTool(tool="web_search", args={}, reason="search", confidence=0.9),
        ]
        result = await _fill_missing_args(ctx, "req-1", predictions, "what are the best python web frameworks", [])
        assert len(result) == 1
        assert result[0].tool == "web_search"
        assert "best python frameworks 2025" in result[0].args["query"]
        ctx.registry.call_free_llm.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_tool_query_extraction_failure_falls_back(self):
        """Search query extraction failure falls back to raw prompt[:200]."""
        ctx = _make_ctx()
        ctx.registry.call_free_llm = AsyncMock(side_effect=RuntimeError("LLM down"))
        user_prompt = "what are the best python web frameworks"
        predictions = [
            AnticipatedTool(tool="web_search", args={}, reason="search", confidence=0.9),
        ]
        result = await _fill_missing_args(ctx, "req-1", predictions, user_prompt, [])
        assert len(result) == 1
        assert result[0].args["query"] == user_prompt[:200]

    @pytest.mark.asyncio
    async def test_grep_tool_without_pattern_skips(self):
        """Grep tool without pattern is skipped."""
        ctx = _make_ctx()
        predictions = [
            AnticipatedTool(tool="grep", args={}, reason="search", confidence=0.9),
        ]
        result = await _fill_missing_args(ctx, "req-1", predictions, "find something", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_tool_with_args_already_filled_passes_through(self):
        """Tool with args already filled passes through unchanged."""
        ctx = _make_ctx()
        predictions = [
            AnticipatedTool(
                tool="read_file",
                args={"path": "config.py"},
                reason="read config",
                confidence=0.9,
            ),
        ]
        result = await _fill_missing_args(ctx, "req-1", predictions, "check config.py", [])
        assert len(result) == 1
        assert result[0].tool == "read_file"
        assert result[0].args["path"] == "config.py"

    @pytest.mark.asyncio
    async def test_list_dir_alias_defaults_to_dot(self):
        """list_files (alias for list_directory) also defaults path to '.'."""
        ctx = _make_ctx()
        predictions = [
            AnticipatedTool(tool="Glob", args={}, reason="list", confidence=0.9),
        ]
        result = await _fill_missing_args(ctx, "req-1", predictions, "show directory", [])
        assert len(result) == 1
        assert result[0].args["path"] == "."

    @pytest.mark.asyncio
    async def test_search_uses_project_context_from_system_msg(self):
        """Search query extraction uses project context from system messages."""
        ctx = _make_ctx()
        ctx.registry.call_free_llm = AsyncMock(return_value='["smartsplit proxy setup"]')
        messages = [{"role": "system", "content": "This is SmartSplit, a proxy anticipateur."}]
        predictions = [
            AnticipatedTool(tool="web_search", args={}, reason="search", confidence=0.9),
        ]
        result = await _fill_missing_args(ctx, "req-1", predictions, "how do I set this up", messages)
        assert len(result) == 1
        # Verify the LLM was called (context extraction happened)
        call_args = ctx.registry.call_free_llm.call_args[0][0]
        assert "SmartSplit" in call_args

    @pytest.mark.asyncio
    async def test_grep_with_pattern_passes_through(self):
        """Grep tool with pattern filled passes through."""
        ctx = _make_ctx()
        predictions = [
            AnticipatedTool(tool="grep", args={"pattern": "TODO"}, reason="find todos", confidence=0.9),
        ]
        result = await _fill_missing_args(ctx, "req-1", predictions, "find all TODOs", [])
        assert len(result) == 1
        assert result[0].args["pattern"] == "TODO"


# ── anticipate_anthropic ────────────────────────────────────────


class TestAnticipateAnthropic:
    @pytest.mark.asyncio
    async def test_returns_empty_when_detector_is_none(self):
        """Returns empty list when ctx.detector is None."""
        ctx = _make_ctx(detector=None)
        result = await anticipate_anthropic(ctx, "req-1", {"messages": []})
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_anticipator_is_none(self):
        """Returns empty list when ctx.anticipator is None."""
        ctx = _make_ctx(anticipator=None)
        result = await anticipate_anthropic(ctx, "req-1", {"messages": []})
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_confidence_below_threshold(self):
        """Returns empty list when prediction confidence is below threshold."""
        ctx = _make_ctx()
        ctx.detector.predict = AsyncMock(return_value=Prediction(should_anticipate=False, confidence=0.3))
        body = {
            "messages": [{"role": "user", "content": "hello"}],
        }
        with (
            patch("smartsplit.proxy.formats.anthropic_messages_to_openai", return_value={"messages": body["messages"]}),
            patch("smartsplit.proxy.formats.extract_anthropic_prompt", return_value="hello"),
        ):
            result = await anticipate_anthropic(ctx, "req-1", body)
        assert result == []
        assert ctx.anticipation_stats["predictions_skipped"] == 1

    @pytest.mark.asyncio
    async def test_happy_path_returns_context_parts(self):
        """Happy path: converts body, predicts tools, executes, returns context parts."""
        tool = AnticipatedTool(tool="read_file", args={"path": "main.py"}, reason="read", confidence=0.9)
        prediction = _prediction([tool])

        ctx = _make_ctx()
        ctx.detector.predict = AsyncMock(return_value=prediction)
        ctx.anticipator.execute = AsyncMock(
            return_value=[
                ToolResult(
                    tool="read_file",
                    args={"path": "main.py"},
                    content="import os\n",
                    success=True,
                    tokens_estimate=3,
                )
            ]
        )

        body = {
            "messages": [{"role": "user", "content": "read main.py"}],
        }
        openai_body = {"messages": [{"role": "user", "content": "read main.py"}]}

        with (
            patch("smartsplit.proxy.formats.anthropic_messages_to_openai", return_value=openai_body),
            patch("smartsplit.proxy.formats.extract_anthropic_prompt", return_value="read main.py"),
        ):
            result = await anticipate_anthropic(ctx, "req-1", body)

        assert len(result) == 1
        assert result[0]["tool"] == "read_file"
        assert "import os" in result[0]["content"]
        assert ctx.anticipation_stats["predictions_made"] == 1
        assert ctx.anticipation_stats["tools_anticipated"] == 1

    @pytest.mark.asyncio
    async def test_handles_conversion_failure_gracefully(self):
        """Handles Anthropic message conversion failure gracefully."""
        ctx = _make_ctx()

        with patch(
            "smartsplit.proxy.formats.anthropic_messages_to_openai",
            side_effect=AttributeError("bad format"),
        ):
            result = await anticipate_anthropic(ctx, "req-1", {"messages": []})

        assert result == []

    @pytest.mark.asyncio
    async def test_filters_out_already_read_paths(self):
        """Filters out paths that were already read in the conversation."""
        tool = AnticipatedTool(tool="read_file", args={"path": "already.py"}, reason="read", confidence=0.9)
        prediction = _prediction([tool])

        ctx = _make_ctx()
        ctx.detector.predict = AsyncMock(return_value=prediction)

        # Build messages where already.py was already read
        messages_with_read = [
            {"role": "user", "content": "read already.py"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "already.py"}),
                        },
                    }
                ],
            },
            {"role": "tool", "content": "file content here", "tool_call_id": "tc_1"},
            {"role": "user", "content": "now explain it"},
        ]
        openai_body = {"messages": messages_with_read}

        with (
            patch("smartsplit.proxy.formats.anthropic_messages_to_openai", return_value=openai_body),
            patch("smartsplit.proxy.formats.extract_anthropic_prompt", return_value="now explain it"),
        ):
            result = await anticipate_anthropic(ctx, "req-1", {"messages": messages_with_read})

        assert result == []

    @pytest.mark.asyncio
    async def test_records_prediction_to_pattern_learner(self):
        """Records prediction to pattern_learner when present."""
        tool = AnticipatedTool(tool="read_file", args={"path": "new.py"}, reason="read", confidence=0.9)
        prediction = _prediction([tool])

        learner = MagicMock()
        ctx = _make_ctx(pattern_learner=learner)
        ctx.detector.predict = AsyncMock(return_value=prediction)
        ctx.anticipator.execute = AsyncMock(
            return_value=[
                ToolResult(
                    tool="read_file",
                    args={"path": "new.py"},
                    content="x = 1\n",
                    success=True,
                    tokens_estimate=2,
                )
            ]
        )

        body = {"messages": [{"role": "user", "content": "read new.py"}]}
        openai_body = {"messages": body["messages"]}

        with (
            patch("smartsplit.proxy.formats.anthropic_messages_to_openai", return_value=openai_body),
            patch("smartsplit.proxy.formats.extract_anthropic_prompt", return_value="read new.py"),
        ):
            result = await anticipate_anthropic(ctx, "req-1", body)

        assert len(result) == 1
        learner.record_prediction.assert_called_once()
        call_kwargs = learner.record_prediction.call_args[1]
        assert call_kwargs["request_id"] == "req-1"
        assert call_kwargs["predicted_tools"][0]["tool"] == "read_file"

    @pytest.mark.asyncio
    async def test_failed_tool_results_not_included(self):
        """Failed tool executions are not included in context parts."""
        tool = AnticipatedTool(tool="read_file", args={"path": "missing.py"}, reason="read", confidence=0.9)
        prediction = _prediction([tool])

        ctx = _make_ctx()
        ctx.detector.predict = AsyncMock(return_value=prediction)
        ctx.anticipator.execute = AsyncMock(
            return_value=[
                ToolResult(
                    tool="read_file",
                    args={"path": "missing.py"},
                    content="error: file not found",
                    success=False,
                    tokens_estimate=0,
                )
            ]
        )

        body = {"messages": [{"role": "user", "content": "read missing.py"}]}
        openai_body = {"messages": body["messages"]}

        with (
            patch("smartsplit.proxy.formats.anthropic_messages_to_openai", return_value=openai_body),
            patch("smartsplit.proxy.formats.extract_anthropic_prompt", return_value="read missing.py"),
        ):
            result = await anticipate_anthropic(ctx, "req-1", body)

        assert result == []


# ── anticipate_tools ────────────────────────────────────────────


class TestAnticipateTools:
    @pytest.mark.asyncio
    async def test_returns_empty_when_detector_is_none(self):
        """Returns empty list when ctx.detector is None."""
        ctx = _make_ctx(detector=None)
        from smartsplit.proxy.formats import OpenAIRequest

        parsed = OpenAIRequest(messages=[{"role": "user", "content": "hello"}])
        result = await anticipate_tools(ctx, "req-1", {}, [], parsed)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_anticipator_is_none(self):
        """Returns empty list when ctx.anticipator is None."""
        ctx = _make_ctx(anticipator=None)
        from smartsplit.proxy.formats import OpenAIRequest

        parsed = OpenAIRequest(messages=[{"role": "user", "content": "hello"}])
        result = await anticipate_tools(ctx, "req-1", {}, [], parsed)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_anticipation(self):
        """Returns empty when prediction says not to anticipate."""
        ctx = _make_ctx()
        ctx.detector.predict = AsyncMock(return_value=Prediction(should_anticipate=False, confidence=0.3))
        from smartsplit.proxy.formats import OpenAIRequest

        parsed = OpenAIRequest(
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "read_file"}}],
        )
        body_dict = {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        }
        result = await anticipate_tools(ctx, "req-1", body_dict, [], parsed)
        assert result == []
        assert ctx.anticipation_stats["predictions_skipped"] == 1

    @pytest.mark.asyncio
    async def test_happy_path_returns_context_parts(self):
        """Happy path: predicts, executes, returns context parts."""
        tool = AnticipatedTool(tool="read_file", args={"path": "app.py"}, reason="read", confidence=0.9)
        prediction = _prediction([tool])

        ctx = _make_ctx()
        ctx.detector.predict = AsyncMock(return_value=prediction)
        ctx.anticipator.execute = AsyncMock(
            return_value=[
                ToolResult(
                    tool="read_file",
                    args={"path": "app.py"},
                    content="from flask import Flask\n",
                    success=True,
                    tokens_estimate=5,
                )
            ]
        )

        from smartsplit.proxy.formats import OpenAIRequest

        parsed = OpenAIRequest(
            messages=[{"role": "user", "content": "read app.py"}],
            tools=[{"type": "function", "function": {"name": "read_file"}}],
        )
        body_dict = {
            "messages": [{"role": "user", "content": "read app.py"}],
        }
        result = await anticipate_tools(ctx, "req-1", body_dict, [], parsed)

        assert len(result) == 1
        assert result[0]["tool"] == "read_file"
        assert "Flask" in result[0]["content"]
        assert ctx.anticipation_stats["predictions_made"] == 1

    @pytest.mark.asyncio
    async def test_records_prediction_to_pattern_learner(self):
        """Records prediction to pattern_learner when present."""
        tool = AnticipatedTool(tool="read_file", args={"path": "x.py"}, reason="read", confidence=0.9)
        prediction = _prediction([tool])

        learner = MagicMock()
        ctx = _make_ctx(pattern_learner=learner)
        ctx.detector.predict = AsyncMock(return_value=prediction)
        ctx.anticipator.execute = AsyncMock(
            return_value=[
                ToolResult(
                    tool="read_file",
                    args={"path": "x.py"},
                    content="pass\n",
                    success=True,
                    tokens_estimate=1,
                )
            ]
        )

        from smartsplit.proxy.formats import OpenAIRequest

        parsed = OpenAIRequest(
            messages=[{"role": "user", "content": "read x.py"}],
            tools=[{"type": "function", "function": {"name": "read_file"}}],
        )
        body_dict = {"messages": [{"role": "user", "content": "read x.py"}]}
        result = await anticipate_tools(ctx, "req-1", body_dict, [], parsed)

        assert len(result) == 1
        learner.record_prediction.assert_called_once()

    @pytest.mark.asyncio
    async def test_filters_already_read_paths(self):
        """Filters out paths already read in the conversation."""
        tool = AnticipatedTool(tool="read_file", args={"path": "done.py"}, reason="read", confidence=0.9)
        prediction = _prediction([tool])

        ctx = _make_ctx()
        ctx.detector.predict = AsyncMock(return_value=prediction)

        from smartsplit.proxy.formats import OpenAIRequest

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "done.py"}),
                        },
                    }
                ],
            },
            {"role": "tool", "content": "content", "tool_call_id": "tc_1"},
            {"role": "user", "content": "now explain"},
        ]
        parsed = OpenAIRequest(
            messages=messages,
            tools=[{"type": "function", "function": {"name": "read_file"}}],
        )
        body_dict = {"messages": messages}
        result = await anticipate_tools(ctx, "req-1", body_dict, [], parsed)
        assert result == []
