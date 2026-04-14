"""Tests for proxy.py pure/helper functions."""

from __future__ import annotations

import asyncio

import pytest

from smartsplit.proxy import (
    _LLM_HOSTS,
    _SMARTSPLIT_PATHS,
    _build_http_response,
    _log_ratelimit_summary,
    _log_sse_usage,
    _parse_transfer_info,
    _read_http_request,
    _rebuild_http_request,
)

# ── Constants ────────────────────────────────────────────────


class TestConstants:
    def test_llm_hosts_not_empty(self):
        assert len(_LLM_HOSTS) > 0
        assert _LLM_HOSTS.issuperset({"api.anthropic.com"})

    def test_smartsplit_paths(self):
        assert "/v1/messages" in _SMARTSPLIT_PATHS
        assert "/v1/chat/completions" in _SMARTSPLIT_PATHS


# ── _build_http_response ─────────────────────────────────────


class TestBuildHttpResponse:
    def test_200_ok(self):
        resp = _build_http_response(200, {"content-type": "application/json"}, b'{"ok":true}')
        text = resp.decode("utf-8")
        assert text.startswith("HTTP/1.1 200 OK\r\n")
        assert "content-type: application/json" in text
        assert '{"ok":true}' in text

    def test_429_status(self):
        resp = _build_http_response(429, {}, b"rate limited")
        assert b"HTTP/1.1 429 Too Many Requests" in resp
        assert b"rate limited" in resp

    def test_content_length_added(self):
        body = b"hello"
        resp = _build_http_response(200, {}, body)
        assert b"content-length: 5" in resp

    def test_empty_body(self):
        resp = _build_http_response(200, {}, b"")
        assert b"content-length: 0" in resp

    def test_unknown_status(self):
        resp = _build_http_response(999, {}, b"")
        assert b"HTTP/1.1 999 Unknown" in resp

    def test_multiple_headers(self):
        resp = _build_http_response(200, {"x-foo": "bar", "x-baz": "qux"}, b"")
        text = resp.decode("utf-8")
        assert "x-foo: bar" in text
        assert "x-baz: qux" in text


# ── _rebuild_http_request ────────────────────────────────────


class TestRebuildHttpRequest:
    def test_basic(self):
        raw = _rebuild_http_request("POST /v1/messages HTTP/1.1", {"host": "api.anthropic.com"}, b'{"model":"x"}')
        text = raw.decode("utf-8")
        assert text.startswith("POST /v1/messages HTTP/1.1\r\n")
        assert "host: api.anthropic.com" in text
        assert '{"model":"x"}' in text

    def test_adds_content_length_if_missing(self):
        raw = _rebuild_http_request("POST /path HTTP/1.1", {}, b"body")
        assert b"content-length: 4" in raw

    def test_no_content_length_for_empty_body(self):
        raw = _rebuild_http_request("GET / HTTP/1.1", {}, b"")
        assert b"content-length" not in raw

    def test_preserves_existing_content_length(self):
        raw = _rebuild_http_request("POST / HTTP/1.1", {"content-length": "10"}, b"0123456789")
        text = raw.decode("utf-8")
        assert text.count("content-length") == 1


# ── _parse_transfer_info ─────────────────────────────────────


class TestParseTransferInfo:
    def test_content_length(self):
        cl, chunked = _parse_transfer_info("HTTP/1.1 200 OK\r\nContent-Length: 42\r\n\r\n")
        assert cl == 42
        assert chunked is False

    def test_chunked(self):
        cl, chunked = _parse_transfer_info("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
        assert cl == 0
        assert chunked is True

    def test_both(self):
        headers = "HTTP/1.1 200 OK\r\nContent-Length: 100\r\nTransfer-Encoding: chunked\r\n\r\n"
        cl, chunked = _parse_transfer_info(headers)
        assert cl == 100
        assert chunked is True

    def test_empty(self):
        cl, chunked = _parse_transfer_info("")
        assert cl == 0
        assert chunked is False

    def test_case_insensitive(self):
        cl, chunked = _parse_transfer_info("content-length: 7\r\ntransfer-encoding: CHUNKED\r\n")
        assert cl == 7
        assert chunked is True


# ── _log_sse_usage ───────────────────────────────────────────


class TestLogSseUsage:
    def test_valid_sse_chunk(self):
        data = b'data: {"type":"message_delta","usage":{"input_tokens":10,"output_tokens":5}}\n'
        _log_sse_usage(data)  # should not raise

    def test_with_cache_tokens(self):
        data = b'data: {"usage":{"input_tokens":10,"cache_read_input_tokens":5,"cache_creation_input_tokens":2,"output_tokens":3}}\n'
        _log_sse_usage(data)

    def test_no_usage(self):
        _log_sse_usage(b'data: {"type":"content_block_start"}\n')

    def test_invalid_json(self):
        _log_sse_usage(b"data: not json\n")

    def test_non_data_lines(self):
        _log_sse_usage(b"event: message_start\n")

    def test_empty(self):
        _log_sse_usage(b"")

    def test_nested_message_usage(self):
        data = b'data: {"message":{"usage":{"input_tokens":100,"output_tokens":50}}}\n'
        _log_sse_usage(data)


# ── _log_ratelimit_summary ───────────────────────────────────


class TestLogRatelimitSummary:
    def test_with_ratelimit_headers(self):
        header = (
            "HTTP/1.1 200 OK\r\n"
            "anthropic-ratelimit-5h-utilization: 0.5\r\n"
            "anthropic-ratelimit-7d-utilization: 0.3\r\n"
            "anthropic-ratelimit-unified-status: active\r\n"
        )
        _log_ratelimit_summary(header)  # should not raise

    def test_no_ratelimit_headers(self):
        _log_ratelimit_summary("HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\n")

    def test_empty(self):
        _log_ratelimit_summary("")


# ── _read_http_request ───────────────────────────────────────


class TestReadHttpRequest:
    @pytest.mark.asyncio
    async def test_basic_request(self):
        reader = asyncio.StreamReader()
        reader.feed_data(
            b'POST /v1/messages HTTP/1.1\r\nhost: api.anthropic.com\r\ncontent-length: 13\r\n\r\n{"model":"x"}'
        )
        reader.feed_eof()

        request_line, headers, body = await _read_http_request(reader)
        assert request_line == "POST /v1/messages HTTP/1.1"
        assert headers["host"] == "api.anthropic.com"
        assert headers["content-length"] == "13"
        assert body == b'{"model":"x"}'

    @pytest.mark.asyncio
    async def test_no_body(self):
        reader = asyncio.StreamReader()
        reader.feed_data(b"GET /health HTTP/1.1\r\nhost: localhost\r\n\r\n")
        reader.feed_eof()

        request_line, headers, body = await _read_http_request(reader)
        assert request_line == "GET /health HTTP/1.1"
        assert body == b""

    @pytest.mark.asyncio
    async def test_multiple_headers(self):
        reader = asyncio.StreamReader()
        reader.feed_data(
            b"POST / HTTP/1.1\r\nhost: localhost\r\nx-api-key: sk-test\r\ncontent-type: application/json\r\n\r\n"
        )
        reader.feed_eof()

        _, headers, _ = await _read_http_request(reader)
        assert headers["x-api-key"] == "sk-test"
        assert headers["content-type"] == "application/json"
