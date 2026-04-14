"""SmartSplit mitmproxy addon — intercepts Claude Code traffic to anticipate tool calls.

Usage:
    # Install mitmproxy
    brew install mitmproxy

    # Trust the mitmproxy CA (first time only)
    mitmproxy  # start once, quit, then:
    security add-trusted-cert -d -r trustRoot ~/.mitmproxy/mitmproxy-ca-cert.pem

    # Launch with SmartSplit addon
    mitmdump -s smartsplit/mitm_addon.py -p 8420

    # Launch Claude Code through the proxy
    NODE_EXTRA_CA_CERTS=~/.mitmproxy/mitmproxy-ca-cert.pem HTTPS_PROXY=http://localhost:8420 claude

How it works:
    1. Claude Code sends a request to api.anthropic.com (thinks it's direct)
    2. mitmproxy intercepts, SmartSplit addon reads the request
    3. If tool calls are predicted with high confidence:
       → Respond DIRECTLY with fake tool_use (no forward to Anthropic)
       → Claude Code executes the reads locally
       → Next request has the file contents in conversation
       → Forward to Anthropic — Claude has everything, fewer round-trips
    4. If no prediction or low confidence:
       → Forward to Anthropic normally (passthrough)
       → Optionally inject anticipated context into system prompt
"""

from __future__ import annotations

import json
import logging

from mitmproxy import http

from smartsplit.intercept import (
    build_fake_response,
    build_sse_response,
    compress_tool_results_in_body,
    extract_tool_names,
    extract_user_prompt,
    has_tool_results,
    predict_reads,
)

logger = logging.getLogger("smartsplit.mitm")

_ANTHROPIC_HOST = "api.anthropic.com"
_MESSAGES_PATH = "/v1/messages"


# ── Main Addon ─────────────────────────────────────────────────


class SmartSplitAddon:
    """mitmproxy addon that intercepts Claude Code traffic for tool anticipation.

    Two modes:
    - ANTICIPATE: predict reads, respond with fake tool_use, let agent execute
    - PASSTHROUGH: forward to Anthropic normally (optionally inject context)
    """

    def __init__(self) -> None:
        # Track sessions where we've sent fake tool calls
        # Key: session_id, Value: True if we faked and are waiting for results
        self._pending_fakes: dict[str, bool] = {}
        self._stats = {
            "requests_total": 0,
            "requests_anticipated": 0,
            "requests_passthrough": 0,
            "requests_compressed": 0,
            "tools_faked": 0,
            "tool_results_compressed": 0,
            "chars_saved": 0,
        }
        logger.info("SmartSplit mitmproxy addon loaded")

    def request(self, flow: http.HTTPFlow) -> None:
        """Intercept requests to api.anthropic.com/v1/messages."""
        # Only intercept Anthropic API calls
        if _ANTHROPIC_HOST not in flow.request.pretty_host:
            return
        if _MESSAGES_PATH not in flow.request.path:
            return

        self._stats["requests_total"] += 1

        try:
            body = flow.request.json()
        except Exception:
            return  # Not JSON, pass through

        tools = body.get("tools", [])
        is_streaming = body.get("stream", False)
        session_id = flow.request.headers.get("x-claude-code-session-id", "unknown")
        model = body.get("model", "claude-sonnet-4-20250514")

        # If no tools, this isn't an agent request — pass through
        if not tools:
            return

        # If this request has tool_results and we have a pending fake → forward to Anthropic
        # (the agent executed our faked reads, now send everything to Claude)
        if has_tool_results(body) and self._pending_fakes.get(session_id):
            self._pending_fakes[session_id] = False
            logger.info("[%s] Tool results received, forwarding to Claude (enriched)", session_id[:8])
            # Compress large tool results before forwarding (Tool-Aware Proxy)
            body, num_compressed = compress_tool_results_in_body(body)
            if num_compressed:
                flow.request.set_content(json.dumps(body).encode("utf-8"))
                self._stats["requests_compressed"] += 1
                self._stats["tool_results_compressed"] += num_compressed
            self._stats["requests_passthrough"] += 1
            return  # Let it pass through to Anthropic

        # If we already have a pending fake, don't fake again — pass through
        if self._pending_fakes.get(session_id):
            return

        # ── Tool-Aware Proxy: compress large tool results on ALL requests ──
        # Even without fake tool calls, compress smart tool results to save tokens
        if has_tool_results(body):
            body, num_compressed = compress_tool_results_in_body(body)
            if num_compressed:
                flow.request.set_content(json.dumps(body).encode("utf-8"))
                self._stats["requests_compressed"] += 1
                self._stats["tool_results_compressed"] += num_compressed
                logger.info("[%s] Compressed %d tool result(s) before forwarding", session_id[:8], num_compressed)

        # Extract prompt and predict reads
        prompt = extract_user_prompt(body)
        if not prompt or len(prompt) < 10:
            return

        tool_names = extract_tool_names(body)
        predictions = predict_reads(prompt, tool_names)

        if not predictions:
            self._stats["requests_passthrough"] += 1
            return  # Nothing to anticipate

        # Check predictions use tools the agent actually has
        valid_predictions = [p for p in predictions if p["tool"] in tool_names]
        if not valid_predictions:
            self._stats["requests_passthrough"] += 1
            return

        # Build fake response with tool_use blocks
        fake_body = build_fake_response(valid_predictions, model=model)

        logger.info(
            "[%s] ANTICIPATING %d tool(s): %s",
            session_id[:8],
            len(valid_predictions),
            [p["tool"] + "(" + str(list(p["input"].values())[0]) + ")" for p in valid_predictions],
        )

        # Respond directly — no forward to Anthropic
        if is_streaming:
            flow.response = http.Response.make(
                200,
                build_sse_response(fake_body),
                {
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            flow.response = http.Response.make(
                200,
                json.dumps(fake_body).encode("utf-8"),
                {"Content-Type": "application/json"},
            )

        self._pending_fakes[session_id] = True
        self._stats["requests_anticipated"] += 1
        self._stats["tools_faked"] += len(valid_predictions)

    def response(self, flow: http.HTTPFlow) -> None:
        """Log responses from Anthropic (passthrough mode)."""
        if _ANTHROPIC_HOST not in flow.request.pretty_host:
            return
        if flow.response and flow.response.status_code == 200:
            session_id = flow.request.headers.get("x-claude-code-session-id", "unknown")
            logger.info("[%s] Anthropic responded (passthrough)", session_id[:8])


# Register the addon
addons = [SmartSplitAddon()]
