"""Tests for smartsplit.intercept — pure functions, no mitmproxy dependency."""

from __future__ import annotations

import json

from smartsplit.intercept import (
    build_fake_response,
    build_sse_response,
    compress_tool_result,
    compress_tool_results_in_body,
    extract_tool_names,
    extract_user_prompt,
    has_tool_results,
    identify_tool_from_result,
    predict_reads,
)

# ── extract_user_prompt ──────────────────────────────────────


class TestExtractUserPrompt:
    def test_string_content(self):
        body = {"messages": [{"role": "user", "content": "fix auth.py"}]}
        assert extract_user_prompt(body) == "fix auth.py"

    def test_list_content_with_text_block(self):
        body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "read the file"}]}]}
        assert extract_user_prompt(body) == "read the file"

    def test_list_content_with_string_block(self):
        body = {"messages": [{"role": "user", "content": ["hello world"]}]}
        assert extract_user_prompt(body) == "hello world"

    def test_picks_last_user_message(self):
        body = {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second"},
            ]
        }
        assert extract_user_prompt(body) == "second"

    def test_no_user_message(self):
        body = {"messages": [{"role": "assistant", "content": "hello"}]}
        assert extract_user_prompt(body) == ""

    def test_empty_messages(self):
        assert extract_user_prompt({"messages": []}) == ""
        assert extract_user_prompt({}) == ""

    def test_list_content_with_non_text_block(self):
        body = {"messages": [{"role": "user", "content": [{"type": "image", "source": "x"}]}]}
        assert extract_user_prompt(body) == ""


# ── extract_tool_names ───────────────────────────────────────


class TestExtractToolNames:
    def test_extracts_names(self):
        body = {"tools": [{"name": "Read"}, {"name": "Write"}, {"name": "Grep"}]}
        assert extract_tool_names(body) == {"Read", "Write", "Grep"}

    def test_empty_tools(self):
        assert extract_tool_names({"tools": []}) == set()
        assert extract_tool_names({}) == set()

    def test_skips_empty_names(self):
        body = {"tools": [{"name": "Read"}, {"name": ""}, {"description": "no name"}]}
        assert extract_tool_names(body) == {"Read"}


# ── has_tool_results ─────────────────────────────────────────


class TestHasToolResults:
    def test_with_tool_result(self):
        body = {
            "messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "data"}]}]
        }
        assert has_tool_results(body) is True

    def test_without_tool_result(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}
        assert has_tool_results(body) is False

    def test_empty(self):
        assert has_tool_results({"messages": []}) is False
        assert has_tool_results({}) is False

    def test_non_list_content(self):
        body = {"messages": [{"role": "user", "content": "just text"}]}
        assert has_tool_results(body) is False


# ── predict_reads ────────────────────────────────────────────


class TestPredictReads:
    def test_file_mentioned_with_Read(self):
        preds = predict_reads("fix the bug in proxy.py", {"Read", "Write"})
        assert any(p["tool"] == "Read" and p["input"]["file_path"] == "proxy.py" for p in preds)

    def test_uses_read_file_fallback(self):
        preds = predict_reads("look at proxy.py", {"read_file", "write_file"})
        assert any(p["tool"] == "read_file" for p in preds)

    def test_test_intent_adds_test_file(self):
        preds = predict_reads("test proxy.py", {"Read", "Write"})
        paths = [p["input"].get("file_path") for p in preds]
        assert "proxy.py" in paths
        assert "test_proxy.py" in paths

    def test_no_files_mentioned(self):
        assert predict_reads("explain the architecture", {"Read"}) == []

    def test_caps_at_3(self):
        preds = predict_reads("read a.py b.py c.py d.py e.py", {"Read"})
        assert len(preds) <= 3

    def test_deduplicates_files(self):
        preds = predict_reads("fix proxy.py and also proxy.py", {"Read"})
        file_paths = [p["input"]["file_path"] for p in preds]
        assert file_paths.count("proxy.py") == 1

    def test_non_py_file_no_test_added(self):
        preds = predict_reads("test config.json", {"Read"})
        paths = [p["input"].get("file_path") for p in preds]
        assert "test_config.py" not in paths


# ── identify_tool_from_result ────────────────────────────────


class TestIdentifyToolFromResult:
    def test_finds_tool_name(self):
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Grep", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "data"}]},
        ]
        assert identify_tool_from_result(messages) == "Grep"

    def test_no_assistant(self):
        assert identify_tool_from_result([{"role": "user", "content": "hi"}]) == ""

    def test_empty(self):
        assert identify_tool_from_result([]) == ""

    def test_assistant_without_tool_use(self):
        messages = [{"role": "assistant", "content": "just text"}]
        assert identify_tool_from_result(messages) == ""

    def test_assistant_list_content_no_tool_use_breaks(self):
        """Assistant with list content but only text blocks hits the break.

        The function should stop at the last assistant with list content,
        even if it has no tool_use block, and return empty string without
        looking further back.
        """
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "old"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "thinking..."}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "data"}]},
        ]
        assert identify_tool_from_result(messages) == ""

    def test_multiple_assistants_returns_last_tool(self):
        """With multiple assistant messages, returns tool from the last one."""
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "old"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "new"}]},
        ]
        assert identify_tool_from_result(messages) == "Bash"

    def test_tool_use_missing_name_returns_empty(self):
        """tool_use block without 'name' key returns empty string."""
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "input": {}}]},
        ]
        assert identify_tool_from_result(messages) == ""

    def test_assistant_with_empty_list_content(self):
        """Assistant with empty list content breaks without finding tool_use."""
        messages = [
            {"role": "assistant", "content": []},
        ]
        assert identify_tool_from_result(messages) == ""


# ── compress_tool_result ─────────────────────────────────────


class TestCompressToolResult:
    def test_bash_large_keeps_head_tail(self):
        content = "\n".join(f"line {i}" for i in range(100))
        result = compress_tool_result(content, "Bash")
        assert "line 0" in result
        assert "line 99" in result
        assert "truncated by SmartSplit" in result

    def test_bash_small_unchanged(self):
        content = "\n".join(f"line {i}" for i in range(20))
        assert compress_tool_result(content, "Bash") == content

    def test_bash_alias(self):
        content = "\n".join(f"line {i}" for i in range(100))
        result = compress_tool_result(content, "bash")
        assert "truncated by SmartSplit" in result

    def test_web_search_truncated(self):
        content = "x" * 5000
        result = compress_tool_result(content, "WebSearch")
        assert len(result) < 1500
        assert "truncated by SmartSplit" in result

    def test_web_search_small_unchanged(self):
        content = "small result"
        assert compress_tool_result(content, "WebSearch") == content

    def test_git_log_large(self):
        content = "\n".join(f"commit {i}" for i in range(80))
        result = compress_tool_result(content, "git_log")
        assert "commit 0" in result
        assert "commit 79" in result
        assert "truncated by SmartSplit" in result

    def test_git_log_small_unchanged(self):
        content = "\n".join(f"commit {i}" for i in range(10))
        assert compress_tool_result(content, "git_log") == content

    def test_default_truncation(self):
        content = "x" * 5000
        result = compress_tool_result(content, "unknown_tool")
        assert "truncated by SmartSplit" in result

    def test_default_small_unchanged(self):
        content = "small"
        assert compress_tool_result(content, "unknown_tool") == content


# ── compress_tool_results_in_body ────────────────────────────


class TestCompressToolResultsInBody:
    def _make_body(self, tool_name: str, content: str) -> dict:
        return {
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": tool_name, "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": content}]},
            ]
        }

    def test_dumb_tool_skipped(self):
        body = self._make_body("Read", "x" * 5000)
        new_body, count = compress_tool_results_in_body(body)
        assert count == 0
        assert new_body is body

    def test_decisional_tool_skipped(self):
        body = self._make_body("Write", "x" * 5000)
        _, count = compress_tool_results_in_body(body)
        assert count == 0

    def test_smart_tool_compressed(self):
        # WebSearch is a SMART_TOOL — large results get compressed
        body = self._make_body("WebSearch", "x" * 5000)
        _, count = compress_tool_results_in_body(body)
        assert count == 1

    def test_small_result_not_compressed(self):
        body = self._make_body("WebSearch", "small")
        _, count = compress_tool_results_in_body(body)
        assert count == 0

    def test_empty_messages(self):
        _, count = compress_tool_results_in_body({"messages": []})
        assert count == 0

    def test_list_content_in_tool_result(self):
        body = {
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "git_log", "input": {}}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [{"type": "text", "text": "x" * 5000}],
                        }
                    ],
                },
            ]
        }
        _, count = compress_tool_results_in_body(body)
        assert count == 1

    def test_list_content_small_not_compressed(self):
        body = {
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "git_log", "input": {}}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "small"}]}
                    ],
                },
            ]
        }
        _, count = compress_tool_results_in_body(body)
        assert count == 0

    def test_non_tool_result_blocks_preserved(self):
        body = {
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "git_diff", "input": {}}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "context"},
                        {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 5000},
                    ],
                },
            ]
        }
        new_body, count = compress_tool_results_in_body(body)
        assert count == 1
        # text block preserved
        assert new_body["messages"][1]["content"][0] == {"type": "text", "text": "context"}

    def test_list_content_with_non_text_sub_block_preserved(self):
        """Non-text sub-blocks (e.g. image) inside list content are preserved (line 246)."""
        body = {
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "web_fetch", "input": {}}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "image", "source": {"data": "base64img"}},
                                {"type": "text", "text": "z" * 5000},
                            ],
                        }
                    ],
                },
            ]
        }
        new_body, count = compress_tool_results_in_body(body)
        assert count == 1
        sub_blocks = new_body["messages"][1]["content"][0]["content"]
        # Image block preserved as-is
        assert sub_blocks[0] == {"type": "image", "source": {"data": "base64img"}}
        # Text block compressed
        assert "truncated by SmartSplit" in sub_blocks[1]["text"]

    def test_at_threshold_not_compressed(self):
        """Content exactly at _COMPRESS_THRESHOLD_CHARS (2000) is NOT compressed."""
        body = self._make_body("WebSearch", "a" * 2000)
        _, count = compress_tool_results_in_body(body)
        assert count == 0

    def test_over_threshold_compressed(self):
        """Content 1 char over threshold IS compressed."""
        body = self._make_body("WebSearch", "a" * 2001)
        _, count = compress_tool_results_in_body(body)
        assert count == 1

    def test_does_not_mutate_original_body(self):
        """Original body dict and message content must not be modified."""
        big_content = "x" * 5000
        body = self._make_body("WebSearch", big_content)
        original_content = body["messages"][1]["content"][0]["content"]
        new_body, count = compress_tool_results_in_body(body)
        assert count == 1
        # Original body still has the uncompressed content
        assert body["messages"][1]["content"][0]["content"] == original_content
        assert len(original_content) == 5000

    def test_body_without_messages_key(self):
        """Body without messages key returns (body, 0) unchanged."""
        body: dict = {"model": "claude-sonnet-4-20250514"}
        new_body, count = compress_tool_results_in_body(body)
        assert count == 0

    def test_non_user_messages_pass_through(self):
        """Assistant messages are not touched, only user messages are scanned."""
        body = {
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "git_log", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "z" * 5000}]},
            ]
        }
        new_body, count = compress_tool_results_in_body(body)
        assert count == 1
        # Assistant message is the exact same object (not copied)
        assert new_body["messages"][0] is body["messages"][0]

    def test_unknown_tool_large_content_compressed(self):
        """Tool not in DUMB_TOOLS or DECISIONAL_TOOLS with large content is compressed."""
        body = self._make_body("some_custom_tool", "u" * 5000)
        _, count = compress_tool_results_in_body(body)
        assert count == 1

    def test_list_content_only_non_text_blocks_no_compression(self):
        """List content with only non-text sub-blocks triggers no compression."""
        body = {
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "web_fetch", "input": {}}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [{"type": "image", "source": {"data": "img"}}],
                        }
                    ],
                },
            ]
        }
        new_body, count = compress_tool_results_in_body(body)
        assert count == 0


# ── build_fake_response ──────────────────────────────────────


class TestBuildFakeResponse:
    def test_structure(self):
        tc = [{"tool": "Read", "input": {"file_path": "a.py"}}, {"tool": "Grep", "input": {"pattern": "TODO"}}]
        resp = build_fake_response(tc)
        assert resp["type"] == "message"
        assert resp["role"] == "assistant"
        assert resp["stop_reason"] == "tool_use"
        assert len(resp["content"]) == 2
        assert resp["content"][0]["name"] == "Read"
        assert resp["content"][1]["name"] == "Grep"
        assert resp["content"][0]["id"] != resp["content"][1]["id"]
        assert resp["content"][0]["id"].startswith("toolu_")

    def test_custom_model(self):
        resp = build_fake_response([], model="claude-opus-4-20250514")
        assert resp["model"] == "claude-opus-4-20250514"

    def test_empty_tool_calls(self):
        resp = build_fake_response([])
        assert resp["content"] == []
        assert resp["stop_reason"] == "tool_use"

    def test_usage_zeros(self):
        resp = build_fake_response([{"tool": "Read", "input": {}}])
        assert resp["usage"]["input_tokens"] == 0
        assert resp["usage"]["output_tokens"] == 0


# ── build_sse_response ───────────────────────────────────────


class TestBuildSseResponse:
    def test_contains_required_events(self):
        fake = build_fake_response([{"tool": "Read", "input": {"file_path": "a.py"}}])
        sse = build_sse_response(fake).decode("utf-8")
        assert "event: message_start" in sse
        assert "event: content_block_start" in sse
        assert "event: content_block_stop" in sse
        assert "event: message_delta" in sse
        assert "event: message_stop" in sse

    def test_empty_content(self):
        fake = build_fake_response([])
        sse = build_sse_response(fake).decode("utf-8")
        assert "event: message_start" in sse
        assert "event: message_stop" in sse
        assert "event: content_block_start" not in sse

    def test_valid_json_in_data_lines(self):
        fake = build_fake_response([{"tool": "Read", "input": {"file_path": "x.py"}}])
        sse = build_sse_response(fake).decode("utf-8")
        for line in sse.split("\n"):
            if line.startswith("data: "):
                json.loads(line[6:])  # must not raise

    def test_multiple_blocks(self):
        fake = build_fake_response(
            [
                {"tool": "Read", "input": {"file_path": "a.py"}},
                {"tool": "Grep", "input": {"pattern": "x"}},
            ]
        )
        sse = build_sse_response(fake).decode("utf-8")
        # Each block has "event: content_block_start\n" + "data: ...content_block_start..."
        assert sse.count("event: content_block_start") == 2
        assert sse.count("event: content_block_stop") == 2
