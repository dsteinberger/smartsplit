"""Tests for SmartSplit intention detector."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from smartsplit.tools.intention_detector import (
    _NULL_PREDICTION,
    AnticipatedTool,
    IntentionDetector,
    Prediction,
    _extract_last_tool_exchange,
    _extract_tool_names,
    _predict_from_rules,
)

# ── _extract_tool_names ─────────────────────────────────────


class TestExtractToolNames:
    def test_openai_format(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "read_file", "description": "Read a file"},
            },
            {
                "type": "function",
                "function": {"name": "write_file", "description": "Write a file"},
            },
        ]
        names = _extract_tool_names(tools)
        assert names == ["read_file", "write_file"]

    def test_empty_list(self):
        assert _extract_tool_names([]) == []

    def test_none(self):
        assert _extract_tool_names(None) == []

    def test_missing_function_key(self):
        tools = [{"type": "function"}]
        assert _extract_tool_names(tools) == []

    def test_missing_name(self):
        tools = [{"type": "function", "function": {"description": "no name"}}]
        assert _extract_tool_names(tools) == []

    def test_non_dict_entries_skipped(self):
        tools = ["not_a_dict", {"type": "function", "function": {"name": "grep"}}]
        names = _extract_tool_names(tools)
        assert names == ["grep"]


# ── _extract_last_tool_exchange ──────────────────────────────


class TestExtractLastToolExchange:
    def test_valid_exchange(self):
        messages = [
            {"role": "user", "content": "Read auth.py"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "auth.py"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "def login(): pass"},
        ]
        name, args, result = _extract_last_tool_exchange(messages)
        assert name == "read_file"
        assert args == {"path": "auth.py"}
        assert result == "def login(): pass"

    def test_too_few_messages(self):
        name, args, result = _extract_last_tool_exchange([{"role": "tool", "content": "x"}])
        assert name == ""
        assert args == {}
        assert result == ""

    def test_no_assistant_with_tool_calls(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "content": "result"},
        ]
        name, args, result = _extract_last_tool_exchange(messages)
        assert name == ""

    def test_empty_tool_calls_list(self):
        messages = [
            {"role": "assistant", "tool_calls": []},
            {"role": "tool", "content": "result"},
        ]
        name, args, result = _extract_last_tool_exchange(messages)
        assert name == ""

    def test_malformed_arguments_json(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "grep", "arguments": "not json"}}],
            },
            {"role": "tool", "content": "matches"},
        ]
        name, args, result = _extract_last_tool_exchange(messages)
        assert name == "grep"
        assert args == {}
        assert result == "matches"

    def test_arguments_as_dict(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "foo.py"}}}],
            },
            {"role": "tool", "content": "code here"},
        ]
        name, args, result = _extract_last_tool_exchange(messages)
        assert name == "read_file"
        assert args == {"path": "foo.py"}

    def test_picks_last_tool_call_from_assistant(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "grep", "arguments": '{"pattern": "TODO"}'}},
                    {"function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
                ],
            },
            {"role": "tool", "content": "file content"},
        ]
        name, args, result = _extract_last_tool_exchange(messages)
        assert name == "read_file"


# ── Prediction filters non-SAFE_TOOLS ───────────────────────


class TestPredictionFiltersSafeTools:
    @pytest.mark.asyncio
    async def test_non_safe_tools_filtered(self):
        """LLM predicts an unsafe tool (write_file) — it should be filtered out."""
        llm_response = json.dumps(
            {
                "should_anticipate": True,
                "confidence": 0.9,
                "anticipated_tools": [
                    {"tool": "write_file", "args": {}, "reason": "write", "confidence": 0.9},
                    {"tool": "read_file", "args": {"path": "x.py"}, "reason": "read", "confidence": 0.85},
                ],
            }
        )
        registry = MagicMock()
        registry.call_worker_llm = AsyncMock(return_value=llm_response)

        detector = IntentionDetector(registry)
        prediction = await detector.predict(
            [{"role": "user", "content": "Fix the auth module"}],
            available_tools=None,
        )
        assert prediction.should_anticipate
        tool_names = [t.tool for t in prediction.tools]
        assert "write_file" not in tool_names
        assert "read_file" in tool_names

    @pytest.mark.asyncio
    async def test_all_unsafe_returns_null(self):
        """If every predicted tool is unsafe, should return null prediction."""
        llm_response = json.dumps(
            {
                "should_anticipate": True,
                "confidence": 0.9,
                "anticipated_tools": [
                    {"tool": "execute_bash", "args": {}, "reason": "run", "confidence": 0.9},
                ],
            }
        )
        registry = MagicMock()
        registry.call_worker_llm = AsyncMock(return_value=llm_response)

        detector = IntentionDetector(registry)
        prediction = await detector.predict(
            [{"role": "user", "content": "Run the tests"}],
            available_tools=None,
        )
        assert not prediction.should_anticipate


# ── Null prediction on empty messages ───────────────────────


class TestNullPrediction:
    @pytest.mark.asyncio
    async def test_empty_messages(self):
        registry = MagicMock()
        detector = IntentionDetector(registry)
        prediction = await detector.predict([], available_tools=None)
        assert prediction is _NULL_PREDICTION
        assert not prediction.should_anticipate
        assert prediction.confidence == 0.0
        assert prediction.tools == []

    @pytest.mark.asyncio
    async def test_llm_returns_low_confidence(self):
        llm_response = json.dumps(
            {
                "should_anticipate": True,
                "confidence": 0.3,
                "anticipated_tools": [],
            }
        )
        registry = MagicMock()
        registry.call_worker_llm = AsyncMock(return_value=llm_response)

        detector = IntentionDetector(registry)
        prediction = await detector.predict(
            [{"role": "user", "content": "hello"}],
            available_tools=None,
        )
        assert not prediction.should_anticipate

    @pytest.mark.asyncio
    async def test_llm_failure_returns_null(self):
        registry = MagicMock()
        registry.call_worker_llm = AsyncMock(side_effect=Exception("LLM down"))

        detector = IntentionDetector(registry)
        prediction = await detector.predict(
            [{"role": "user", "content": "What time is it?"}],
            available_tools=None,
        )
        # No rules match, LLM fails → null prediction
        assert not prediction.should_anticipate

    @pytest.mark.asyncio
    async def test_unknown_role_returns_null(self):
        registry = MagicMock()
        detector = IntentionDetector(registry)
        prediction = await detector.predict(
            [{"role": "system", "content": "You are an assistant"}],
            available_tools=None,
        )
        assert not prediction.should_anticipate


# ── _predict_from_rules ───────────────────────────────────


class TestPredictFromRules:
    def test_single_file(self):
        prediction = _predict_from_rules("lis proxy.py", set())
        assert prediction.should_anticipate
        assert any(t.tool == "read_file" and t.args["path"] == "proxy.py" for t in prediction.tools)

    def test_multiple_files(self):
        prediction = _predict_from_rules("compare proxy.py et router.py", set())
        assert prediction.should_anticipate
        paths = [t.args.get("path") for t in prediction.tools if t.tool == "read_file"]
        assert "proxy.py" in paths
        assert "router.py" in paths

    def test_no_files_no_intent(self):
        prediction = _predict_from_rules("bonjour comment ca va", set())
        assert not prediction.should_anticipate

    def test_caps_at_3(self):
        prediction = _predict_from_rules("look at a.py b.py c.py d.py e.py", set())
        assert prediction.should_anticipate
        assert len(prediction.tools) <= 3

    def test_stacktrace_extracts_files(self):
        trace = 'File "smartsplit/proxy.py", line 42, in handle\n  File "smartsplit/router.py", line 10'
        prediction = _predict_from_rules("fix this error:\n" + trace, set())
        paths = [t.args.get("path") for t in prediction.tools if t.tool == "read_file"]
        assert "smartsplit/proxy.py" in paths
        assert "smartsplit/router.py" in paths

    def test_search_intent(self):
        prediction = _predict_from_rules("find where the function is defined", set())
        assert prediction.should_anticipate
        assert any(t.tool == "grep" for t in prediction.tools)

    def test_test_intent_adds_test_file(self):
        prediction = _predict_from_rules("write tests for proxy.py", set())
        paths = [t.args.get("path") for t in prediction.tools if t.tool == "read_file"]
        assert "proxy.py" in paths
        assert "test_proxy.py" in paths

    def test_error_intent_without_file(self):
        prediction = _predict_from_rules("fix the TypeError in the code", set())
        assert prediction.should_anticipate
        assert any(t.tool == "grep" for t in prediction.tools)

    def test_read_tool_remaps_to_client_alias(self):
        """When the client exposes ``Read``, rules must emit ``Read``, not ``read_file``."""
        prediction = _predict_from_rules("lis proxy.py", {"Read"})
        assert prediction.should_anticipate
        assert all(t.tool == "Read" for t in prediction.tools if "path" in t.args)

    def test_grep_tool_remaps_to_client_alias(self):
        """When the client exposes ``Grep``, rules must emit ``Grep``, not ``grep``."""
        prediction = _predict_from_rules("find where the function is defined", {"Grep"})
        assert prediction.should_anticipate
        assert any(t.tool == "Grep" for t in prediction.tools)

    def test_stacktrace_remaps_to_client_alias(self):
        trace = 'File "smartsplit/proxy.py", line 42, in handle'
        prediction = _predict_from_rules("fix this error:\n" + trace, {"Read"})
        assert prediction.should_anticipate
        assert all(t.tool == "Read" for t in prediction.tools)

    def test_test_intent_remaps_to_client_alias(self):
        prediction = _predict_from_rules("write tests for proxy.py", {"Read"})
        assert all(t.tool == "Read" for t in prediction.tools)

    def test_joined_paths_never_emitted_as_one(self):
        """Prose notation `a.json/b.json` must never be emitted as a single concatenated path."""
        prediction = _predict_from_rules("compare package.json/tsconfig.json for config", set())
        paths = [t.args.get("path") for t in prediction.tools if t.tool == "read_file"]
        assert "package.json/tsconfig.json" not in paths
        assert "package.json" in paths
        assert "tsconfig.json" in paths

    def test_real_subpath_not_split(self):
        """`src/main.py` is a real path — must stay intact, not be split on `/`."""
        prediction = _predict_from_rules("read src/main.py please", set())
        paths = [t.args.get("path") for t in prediction.tools if t.tool == "read_file"]
        assert "src/main.py" in paths


# ── _merge_all ─────────────────────────────────────────────


class TestMergeAll:
    def _make_detector(self) -> IntentionDetector:
        registry = MagicMock()
        return IntentionDetector(registry)

    def test_deduplicates(self):
        detector = self._make_detector()
        rule = Prediction(
            should_anticipate=True,
            confidence=0.95,
            tools=[
                AnticipatedTool(tool="read_file", args={"path": "proxy.py"}, reason="file mention", confidence=0.95),
            ],
        )
        llm = Prediction(
            should_anticipate=True,
            confidence=0.85,
            tools=[
                AnticipatedTool(tool="read_file", args={"path": "proxy.py"}, reason="llm predicted", confidence=0.85),
            ],
        )
        merged = detector._merge_all(rule, llm, [])
        # Deduplicated: same tool + same path -> only 1
        read_proxy = [t for t in merged.tools if t.tool == "read_file" and t.args.get("path") == "proxy.py"]
        assert len(read_proxy) == 1

    def test_combines_sources(self):
        detector = self._make_detector()
        rule = Prediction(
            should_anticipate=True,
            confidence=0.95,
            tools=[
                AnticipatedTool(tool="read_file", args={"path": "proxy.py"}, reason="file mention", confidence=0.95),
            ],
        )
        llm = Prediction(
            should_anticipate=True,
            confidence=0.8,
            tools=[
                AnticipatedTool(tool="grep", args={"pattern": "TODO"}, reason="llm", confidence=0.8),
            ],
        )
        patterns = [{"tool": "list_directory", "args": {"path": "."}, "confidence": 0.75, "source": "sequential"}]
        merged = detector._merge_all(rule, llm, patterns)
        tool_names = [t.tool for t in merged.tools]
        assert "read_file" in tool_names
        assert "grep" in tool_names
        assert "list_directory" in tool_names

    def test_rules_priority(self):
        """Rule prediction appears first (highest confidence)."""
        detector = self._make_detector()
        rule = Prediction(
            should_anticipate=True,
            confidence=0.95,
            tools=[
                AnticipatedTool(tool="read_file", args={"path": "auth.py"}, reason="file mention", confidence=0.95),
            ],
        )
        llm = Prediction(
            should_anticipate=True,
            confidence=0.8,
            tools=[
                AnticipatedTool(tool="grep", args={"pattern": "TODO"}, reason="llm", confidence=0.8),
            ],
        )
        merged = detector._merge_all(rule, llm, [])
        # Rule tool has highest confidence (0.95 > 0.8), should be first after sort
        assert merged.tools[0].tool == "read_file"
        assert merged.tools[0].confidence == 0.95

    def test_pattern_write_tool_is_filtered(self):
        """Pattern learner may have observed write tools — they must never leak into predictions."""
        detector = self._make_detector()
        rule = Prediction(should_anticipate=False, confidence=0.0, tools=[])
        llm = Prediction(should_anticipate=False, confidence=0.0, tools=[])
        patterns = [
            {"tool": "Write", "args": {"file_path": "x.py", "content": "..."}, "confidence": 0.95, "source": "seq"},
            {"tool": "Bash", "args": {"command": "rm -rf /"}, "confidence": 0.95, "source": "seq"},
            {"tool": "Read", "args": {"file_path": "x.py"}, "confidence": 0.9, "source": "seq"},
        ]
        merged = detector._merge_all(rule, llm, patterns)
        tool_names = [t.tool for t in merged.tools]
        assert "Write" not in tool_names
        assert "Bash" not in tool_names
        assert "Read" in tool_names

    def test_final_safety_gate_drops_non_safe_tools(self):
        """Even if a non-safe tool somehow makes it past per-source filters, the global gate catches it."""
        detector = self._make_detector()
        # Simulate a rule prediction bug: a non-safe tool leaks through
        rule = Prediction(
            should_anticipate=True,
            confidence=0.95,
            tools=[
                AnticipatedTool(tool="Bash", args={"command": "ls"}, reason="bug", confidence=0.95),
                AnticipatedTool(tool="Read", args={"path": "x.py"}, reason="rule", confidence=0.9),
            ],
        )
        merged = detector._merge_all(rule, Prediction(should_anticipate=False, confidence=0.0, tools=[]), [])
        tool_names = [t.tool for t in merged.tools]
        assert "Bash" not in tool_names
        assert "Read" in tool_names
