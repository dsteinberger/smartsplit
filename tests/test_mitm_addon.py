"""Tests for SmartSplit mitmproxy addon helper functions.

Tests the pure functions (compression, extraction, prediction) without
requiring mitmproxy itself. The mitmproxy import is skipped if not available.
"""

from __future__ import annotations

import sys

import pytest

# mitm_addon requires mitmproxy which needs Python >= 3.12
pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="mitmproxy requires Python >= 3.12")


@pytest.fixture(autouse=True)
def _check_mitmproxy():
    """Skip all tests in this module if mitmproxy is not installed."""
    pytest.importorskip("mitmproxy")


# ── extract_user_prompt ──────────────────────────────────────


class TestExtractUserPrompt:
    def test_string_content(self):
        from smartsplit.intercept import extract_user_prompt

        body = {"messages": [{"role": "user", "content": "fix auth.py"}]}
        assert extract_user_prompt(body) == "fix auth.py"

    def test_list_content_with_text_block(self):
        from smartsplit.intercept import extract_user_prompt

        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "read the file"}],
                }
            ]
        }
        assert extract_user_prompt(body) == "read the file"

    def test_picks_last_user_message(self):
        from smartsplit.intercept import extract_user_prompt

        body = {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second"},
            ]
        }
        assert extract_user_prompt(body) == "second"

    def test_no_user_message(self):
        from smartsplit.intercept import extract_user_prompt

        body = {"messages": [{"role": "assistant", "content": "hello"}]}
        assert extract_user_prompt(body) == ""

    def test_empty_messages(self):
        from smartsplit.intercept import extract_user_prompt

        assert extract_user_prompt({"messages": []}) == ""
        assert extract_user_prompt({}) == ""


# ── extract_tool_names ───────────────────────────────────────


class TestExtractToolNames:
    def test_extracts_names(self):
        from smartsplit.intercept import extract_tool_names

        body = {
            "tools": [
                {"name": "Read", "description": "read a file"},
                {"name": "Write", "description": "write a file"},
                {"name": "Grep", "description": "search"},
            ]
        }
        names = extract_tool_names(body)
        assert names == {"Read", "Write", "Grep"}

    def test_empty_tools(self):
        from smartsplit.intercept import extract_tool_names

        assert extract_tool_names({"tools": []}) == set()
        assert extract_tool_names({}) == set()


# ── has_tool_results ─────────────────────────────────────────


class TestHasToolResults:
    def test_with_tool_result(self):
        from smartsplit.intercept import has_tool_results

        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_123", "content": "file contents"},
                    ],
                }
            ]
        }
        assert has_tool_results(body) is True

    def test_without_tool_result(self):
        from smartsplit.intercept import has_tool_results

        body = {"messages": [{"role": "user", "content": "hello"}]}
        assert has_tool_results(body) is False

    def test_empty_messages(self):
        from smartsplit.intercept import has_tool_results

        assert has_tool_results({"messages": []}) is False


# ── predict_reads ────────────────────────────────────────────


class TestPredictReads:
    def test_file_mentioned(self):
        from smartsplit.intercept import predict_reads

        predictions = predict_reads("fix the bug in proxy.py", {"Read", "Write", "Grep"})
        assert len(predictions) >= 1
        assert any(p["tool"] == "Read" and p["input"]["file_path"] == "proxy.py" for p in predictions)

    def test_uses_read_file_when_no_Read(self):
        from smartsplit.intercept import predict_reads

        predictions = predict_reads("look at proxy.py", {"read_file", "write_file"})
        assert any(p["tool"] == "read_file" for p in predictions)

    def test_test_intent_adds_test_file(self):
        from smartsplit.intercept import predict_reads

        predictions = predict_reads("test proxy.py", {"Read", "Write"})
        paths = [p["input"].get("file_path") for p in predictions]
        assert "proxy.py" in paths
        assert "test_proxy.py" in paths

    def test_no_files_mentioned(self):
        from smartsplit.intercept import predict_reads

        predictions = predict_reads("explain how the code works", {"Read", "Write"})
        assert predictions == []

    def test_caps_at_3(self):
        from smartsplit.intercept import predict_reads

        predictions = predict_reads("read a.py b.py c.py d.py e.py", {"Read"})
        assert len(predictions) <= 3

    def test_short_prompt_no_prediction(self):
        from smartsplit.intercept import predict_reads

        # In the addon, prompts < 10 chars are skipped (tested at addon level)
        # but predict_reads itself doesn't enforce this
        predictions = predict_reads("hi", {"Read"})
        assert predictions == []


# ── _compress_tool_result ─────────────────────────────────────


class TestCompressToolResult:
    def test_grep_passes_through_uncompressed(self):
        """Grep results are now in DUMB_TOOLS — compress_tool_result is never called for them,
        but if it were, it falls through to the default handler."""
        from smartsplit.intercept import compress_tool_result

        content = "\n".join(f"match {i}" for i in range(100))
        result = compress_tool_result(content, "Grep")
        # Grep is not in the compress function's known tools anymore,
        # falls through to default (truncate at 2x target chars)
        assert len(result) <= len(content)

    def test_bash_keeps_head_tail(self):
        from smartsplit.intercept import compress_tool_result

        content = "\n".join(f"line {i}" for i in range(100))
        result = compress_tool_result(content, "Bash")
        assert "line 0" in result  # head
        assert "line 99" in result  # tail
        assert "truncated by SmartSplit" in result

    def test_bash_small_output_unchanged(self):
        from smartsplit.intercept import compress_tool_result

        content = "\n".join(f"line {i}" for i in range(20))
        result = compress_tool_result(content, "Bash")
        assert result == content

    def test_web_search_truncated(self):
        from smartsplit.intercept import compress_tool_result

        content = "x" * 5000
        result = compress_tool_result(content, "WebSearch")
        assert len(result) < 1500
        assert "truncated by SmartSplit" in result

    def test_git_log_keeps_head_tail(self):
        from smartsplit.intercept import compress_tool_result

        content = "\n".join(f"commit {i}" for i in range(80))
        result = compress_tool_result(content, "git_log")
        assert "commit 0" in result
        assert "commit 79" in result
        assert "truncated by SmartSplit" in result


# ── compress_tool_results_in_body ────────────────────────────


class TestCompressToolResultsInBody:
    def _make_body_with_tool_result(self, tool_name: str, content: str) -> dict:
        """Build an Anthropic body with assistant tool_use + user tool_result."""
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": tool_name,
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": content,
                        }
                    ],
                },
            ]
        }

    def test_grep_passes_through_uncompressed(self):
        """Grep results pass through intact — LLM needs full search results."""
        from smartsplit.intercept import compress_tool_results_in_body

        big_content = "\n".join(f"match {i}: some code here" for i in range(200))
        body = self._make_body_with_tool_result("Grep", big_content)
        new_body, count = compress_tool_results_in_body(body)
        assert count == 0

    def test_skips_dumb_tools(self):
        from smartsplit.intercept import compress_tool_results_in_body

        big_content = "x" * 5000
        body = self._make_body_with_tool_result("Read", big_content)
        new_body, count = compress_tool_results_in_body(body)
        assert count == 0
        assert new_body is body  # unchanged

    def test_skips_decisional_tools(self):
        from smartsplit.intercept import compress_tool_results_in_body

        big_content = "x" * 5000
        body = self._make_body_with_tool_result("Write", big_content)
        new_body, count = compress_tool_results_in_body(body)
        assert count == 0

    def test_skips_small_results(self):
        from smartsplit.intercept import compress_tool_results_in_body

        small_content = "just a few lines"
        body = self._make_body_with_tool_result("Grep", small_content)
        new_body, count = compress_tool_results_in_body(body)
        assert count == 0

    def test_empty_messages(self):
        from smartsplit.intercept import compress_tool_results_in_body

        body: dict = {"messages": []}
        new_body, count = compress_tool_results_in_body(body)
        assert count == 0


# ── build_fake_response ──────────────────────────────────────


class TestBuildFakeResponse:
    def test_structure(self):
        from smartsplit.intercept import build_fake_response

        tool_calls = [
            {"tool": "Read", "input": {"file_path": "proxy.py"}},
            {"tool": "Grep", "input": {"pattern": "TODO"}},
        ]
        response = build_fake_response(tool_calls)

        assert response["type"] == "message"
        assert response["role"] == "assistant"
        assert response["stop_reason"] == "tool_use"
        assert len(response["content"]) == 2
        assert response["content"][0]["type"] == "tool_use"
        assert response["content"][0]["name"] == "Read"
        assert response["content"][0]["input"] == {"file_path": "proxy.py"}
        assert response["content"][1]["name"] == "Grep"
        # IDs should be unique
        assert response["content"][0]["id"] != response["content"][1]["id"]
        # IDs should start with "toolu_"
        assert response["content"][0]["id"].startswith("toolu_")

    def test_custom_model(self):
        from smartsplit.intercept import build_fake_response

        response = build_fake_response([], model="claude-opus-4-20250514")
        assert response["model"] == "claude-opus-4-20250514"


# ── build_sse_response ──────────────────────────────────────


class TestBuildSseResponse:
    def test_contains_required_events(self):
        from smartsplit.intercept import build_fake_response, build_sse_response

        fake = build_fake_response([{"tool": "Read", "input": {"file_path": "a.py"}}])
        sse = build_sse_response(fake).decode("utf-8")

        assert "event: message_start" in sse
        assert "event: content_block_start" in sse
        assert "event: content_block_stop" in sse
        assert "event: message_delta" in sse
        assert "event: message_stop" in sse

    def test_empty_content(self):
        from smartsplit.intercept import build_fake_response, build_sse_response

        fake = build_fake_response([])
        sse = build_sse_response(fake).decode("utf-8")

        assert "event: message_start" in sse
        assert "event: message_stop" in sse
        # No content blocks
        assert "event: content_block_start" not in sse


# ── identify_tool_from_result ────────────────────────────────


class TestIdentifyToolFromResult:
    def test_finds_tool_name(self):
        from smartsplit.intercept import identify_tool_from_result

        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_123", "name": "Grep", "input": {"pattern": "TODO"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_123", "content": "matched lines"},
                ],
            },
        ]
        assert identify_tool_from_result(messages) == "Grep"

    def test_no_assistant_message(self):
        from smartsplit.intercept import identify_tool_from_result

        messages = [{"role": "user", "content": "hello"}]
        assert identify_tool_from_result(messages) == ""

    def test_empty_messages(self):
        from smartsplit.intercept import identify_tool_from_result

        assert identify_tool_from_result([]) == ""
