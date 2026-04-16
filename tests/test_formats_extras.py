"""Coverage-focused tests for smartsplit.proxy.formats — pure-logic helpers."""

from __future__ import annotations

import json

from smartsplit.proxy.formats import (
    anthropic_has_tool_result,
    anthropic_messages_to_openai,
    anthropic_to_flat_messages,
    build_fake_openai_tool_response,
    build_response,
    extract_anthropic_prompt,
    openai_response_to_anthropic,
    response_to_sse_chunks,
    strip_agent_metadata,
)


class TestStripAgentMetadata:
    def test_collapses_multiple_blank_lines(self):
        text = "before\n<x>noise</x>\n\n\n\nafter"
        assert strip_agent_metadata(text) == "before\n\nafter"


class TestBuildResponse:
    def test_uses_provider_token_split_when_provided(self):
        resp = build_response("hi", prompt_tokens=12, completion_tokens=3)
        assert resp["usage"]["prompt_tokens"] == 12
        assert resp["usage"]["completion_tokens"] == 3
        assert resp["usage"]["total_tokens"] == 15


class TestResponseToSSEChunks:
    def test_no_choices_falls_back_to_empty_stream(self):
        chunks = response_to_sse_chunks({"choices": []})
        assert chunks[-1] == "data: [DONE]\n\n"
        assert len(chunks) == 4  # stream_chunks default


class TestExtractAnthropicPrompt:
    def test_string_blocks_inside_content_list(self):
        body = {"messages": [{"role": "user", "content": ["raw string block"]}]}
        assert extract_anthropic_prompt(body) == "raw string block"


class TestAnthropicHasToolResult:
    def test_true_when_present(self):
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]},
            ]
        }
        assert anthropic_has_tool_result(body) is True

    def test_false_when_last_user_has_text_only(self):
        body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
        assert anthropic_has_tool_result(body) is False

    def test_false_when_content_is_string(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert anthropic_has_tool_result(body) is False

    def test_false_when_no_user_message(self):
        body = {"messages": [{"role": "assistant", "content": "hello"}]}
        assert anthropic_has_tool_result(body) is False


class TestAnthropicMessagesToOpenAI:
    def test_system_as_block_list(self):
        body = {
            "system": [{"type": "text", "text": "be kind"}, "and clear"],
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = anthropic_messages_to_openai(body)
        assert result["messages"][0] == {"role": "system", "content": "be kind\nand clear"}

    def test_assistant_string_content_passthrough(self):
        body = {"messages": [{"role": "assistant", "content": "raw"}]}
        result = anthropic_messages_to_openai(body)
        assert result["messages"][0] == {"role": "assistant", "content": "raw"}

    def test_assistant_skips_non_dict_blocks(self):
        body = {"messages": [{"role": "assistant", "content": ["ignored", {"type": "text", "text": "kept"}]}]}
        result = anthropic_messages_to_openai(body)
        assert result["messages"][0]["content"] == "kept"

    def test_user_non_string_non_list_falls_back(self):
        body = {"messages": [{"role": "user", "content": None}]}
        result = anthropic_messages_to_openai(body)
        assert result["messages"][0] == {"role": "user", "content": ""}

    def test_user_text_only_block_list(self):
        body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
        result = anthropic_messages_to_openai(body)
        assert result["messages"][0] == {"role": "user", "content": "hi"}

    def test_user_tool_result_skips_non_dict(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        "skip me",
                        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                    ],
                }
            ]
        }
        result = anthropic_messages_to_openai(body)
        assert result["messages"] == [{"role": "tool", "content": "ok", "tool_call_id": "t1"}]

    def test_user_tool_result_emits_text_message_too(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "follow-up"},
                        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                    ],
                }
            ]
        }
        result = anthropic_messages_to_openai(body)
        assert {"role": "tool", "content": "ok", "tool_call_id": "t1"} in result["messages"]
        assert {"role": "user", "content": "follow-up"} in result["messages"]

    def test_unknown_role_passthrough(self):
        body = {"messages": [{"role": "developer", "content": "hi"}]}
        result = anthropic_messages_to_openai(body)
        assert result["messages"][0] == {"role": "developer", "content": "hi"}

    def test_temperature_and_stream_propagated(self):
        body = {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.0, "stream": True}
        result = anthropic_messages_to_openai(body)
        assert result["temperature"] == 0.0
        assert result["stream"] is True

    def test_tool_choice_specific_tool(self):
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "Read", "description": "x", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "Read"},
        }
        result = anthropic_messages_to_openai(body)
        assert result["tool_choice"] == {"type": "function", "function": {"name": "Read"}}

    def test_tool_choice_unknown_type_dropped(self):
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "mystery"},
        }
        assert "tool_choice" not in anthropic_messages_to_openai(body)

    def test_tool_result_content_as_text_block_list(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [{"type": "text", "text": "alpha"}, "beta"],
                        }
                    ],
                }
            ]
        }
        result = anthropic_messages_to_openai(body)
        assert result["messages"][0] == {"role": "tool", "content": "alpha\nbeta", "tool_call_id": "t1"}


class TestOpenAIResponseToAnthropic:
    def test_invalid_json_args_become_empty_dict(self):
        resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "Read", "arguments": "{not json"}},
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        out = openai_response_to_anthropic(resp, model="claude-test")
        tool_blocks = [b for b in out["content"] if b["type"] == "tool_use"]
        assert tool_blocks[0]["input"] == {}
        assert out["stop_reason"] == "tool_use"

    def test_no_choices_emits_empty_text_block(self):
        out = openai_response_to_anthropic({"choices": []}, model="m")
        assert out["content"] == [{"type": "text", "text": ""}]
        assert out["stop_reason"] == "end_turn"

    def test_finish_reason_length_maps_to_max_tokens(self):
        resp = {"choices": [{"message": {"content": "hi"}, "finish_reason": "length"}]}
        out = openai_response_to_anthropic(resp, model="m")
        assert out["stop_reason"] == "max_tokens"


class TestAnthropicToFlatMessages:
    def test_system_as_block_list(self):
        body = {
            "system": [{"type": "text", "text": "be kind"}, "and clear", {"type": "ignored"}],
            "messages": [{"role": "user", "content": "hi"}],
        }
        flat = anthropic_to_flat_messages(body)
        assert flat[0] == {"role": "system", "content": "be kind\nand clear"}

    def test_tool_result_content_as_block_list(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [{"type": "text", "text": "alpha"}, "beta"],
                        }
                    ],
                }
            ]
        }
        flat = anthropic_to_flat_messages(body)
        assert flat[0]["content"] == "alpha\nbeta"

    def test_string_block_in_content_list(self):
        body = {"messages": [{"role": "user", "content": ["plain"]}]}
        flat = anthropic_to_flat_messages(body)
        assert flat[0]["content"] == "plain"

    def test_tool_use_block_rendered_as_marker(self):
        body = {
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]}
            ]
        }
        flat = anthropic_to_flat_messages(body)
        assert "[tool_use: Read]" in flat[0]["content"]

    def test_non_string_non_list_content(self):
        body = {"messages": [{"role": "user", "content": None}]}
        flat = anthropic_to_flat_messages(body)
        assert flat[0] == {"role": "user", "content": ""}


class TestBuildFakeOpenAIToolResponse:
    def test_shape(self):
        out = build_fake_openai_tool_response(
            [{"tool": "Read", "input": {"path": "auth.py"}}],
            model="brain-x",
        )
        assert out["model"] == "brain-x"
        choice = out["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        tc = choice["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "Read"
        assert json.loads(tc["function"]["arguments"]) == {"path": "auth.py"}
        assert tc["id"].startswith("call_smartsplit_")
        assert tc["index"] == 0
