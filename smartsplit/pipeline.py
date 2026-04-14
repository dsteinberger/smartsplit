"""SmartSplit — Free multi-LLM backend with intelligent routing.

An OpenAI-compatible endpoint that routes every request to the best free LLM
for the task. Works as a drop-in backend for Continue, Cline, Aider, or any
OpenAI-compatible client.

Usage:
    smartsplit --port 8420
    # Then point your client at http://localhost:8420/v1
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
import uuid as _uuid
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from smartsplit.anticipation import (
    extract_actual_tool_calls,
)
from smartsplit.config import SmartSplitConfig, load_config
from smartsplit.detector import _LLM_DETECT_MIN_CHARS, TriageDecision, detect, detect_with_llm
from smartsplit.enrichment import (
    enrich_and_forward,
    enrich_only,
)
from smartsplit.formats import (
    OpenAIRequest,
    anthropic_has_tool_named,
    anthropic_has_tool_result,
    anthropic_has_tools,
    anthropic_messages_to_openai,
    anthropic_response_to_sse,
    anthropic_to_flat_messages,
    build_fake_anthropic_tool_response,
    build_fake_openai_tool_response,
    build_response,
    extract_anthropic_prompt,
    extract_prompt,
    inject_anthropic_system_context,
    iter_sse,
    openai_response_to_anthropic,
    response_to_sse_chunks,
    stream_chunks,
    strip_agent_metadata,
)
from smartsplit.intention_detector import IntentionDetector
from smartsplit.learning import BanditScorer
from smartsplit.models import (
    Mode,
    ProviderType,
    RequestLog,
    RouteResult,
    RoutingStep,
    TaskType,
    TerminationState,
    short_id,
)
from smartsplit.planner import Planner
from smartsplit.providers.registry import _PROVIDER_CALL_TIMEOUT, ProviderRegistry
from smartsplit.quota import QuotaTracker
from smartsplit.router import Router
from smartsplit.tool_anticipator import ToolAnticipator
from smartsplit.tool_pattern_learner import ToolPatternLearner

logger = logging.getLogger("smartsplit.proxy")

_MAX_LOG_ENTRIES = 50
_FAKE_TOOL_CONFIDENCE = 0.85  # higher than injection (0.7) — fake responses are riskier

# Enrichment types useful in proxy mode — context_summary is redundant (brain has full context).
_PROXY_USEFUL_ENRICHMENTS = {"web_search", "multi_perspective", "pre_analysis"}

# HTTP hop-by-hop headers to strip when proxying responses.
_HOP_BY_HOP = {"connection", "transfer-encoding", "keep-alive", "upgrade", "content-length", "content-encoding"}


class _AnthropicPassthroughError(Exception):
    """Raised when Anthropic returns a rate limit or auth error that should be propagated as-is."""

    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(str(status_code))


# ── Server context ────────────────────────────────────────────


@dataclass
class ProxyContext:
    """Shared server context holding all components and runtime state."""

    config: SmartSplitConfig
    registry: ProviderRegistry
    planner: Planner
    router: Router
    quota: QuotaTracker
    bandit: BanditScorer
    http: httpx.AsyncClient
    detector: IntentionDetector | None = None
    anticipator: ToolAnticipator | None = None
    pattern_learner: ToolPatternLearner | None = None
    request_logs: deque[RequestLog] = field(default_factory=lambda: deque(maxlen=_MAX_LOG_ENTRIES))
    mode: Mode = Mode.BALANCED
    enabled: bool = True
    triage_counts: dict[str, int] = field(
        default_factory=lambda: {
            TriageDecision.TRANSPARENT: 0,
            TriageDecision.ENRICH: 0,
        }
    )
    # Fake tool_use tracking — session_id → True if we sent a fake and await tool_results
    pending_fakes: dict[str, bool] = field(default_factory=dict)
    # Anticipation stats
    anticipation_stats: dict[str, int] = field(
        default_factory=lambda: {
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
    )
    # Enrichment backoff — skip enrichment until this timestamp (set from retry-after on 429)
    enrichment_skip_until: float = 0.0  # 0 = enrichment allowed


# ── Helpers ────────────────────────────────────────────────────


def _truncate_messages(messages: list[dict[str, str]], max_chars: int) -> list[dict[str, str]]:
    """Truncate conversation to fit within max_chars.

    Keeps system messages + most recent messages, drops old messages from the middle.
    """
    system = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]

    budget = max_chars - sum(len(m["content"]) for m in system)
    kept: list[dict[str, str]] = []
    for m in reversed(non_system):
        if budget - len(m["content"]) < 0:
            break
        kept.append(m)
        budget -= len(m["content"])
    kept.reverse()

    # Always keep at least the last user message (truncated if needed)
    if not kept and non_system:
        last = non_system[-1]
        kept = [{"role": last["role"], "content": last["content"][: max(budget, 200)]}]

    return system + kept


# ── Anthropic pipeline (shared between Starlette handler and TLS proxy) ──


@dataclass
class PipelineResult:
    """Result of processing an Anthropic request through the SmartSplit pipeline."""

    status_code: int = 200
    body: dict | None = None
    body_bytes: bytes = b""
    headers: dict[str, str] = field(default_factory=lambda: {"content-type": "application/json"})
    # For streaming passthrough — an httpx response to iterate over
    streaming_response: object | None = None

    @property
    def is_streaming(self) -> bool:
        return self.streaming_response is not None


async def _run_pipeline(
    ctx: ProxyContext,
    body_dict: dict,
    request_id: str,
) -> dict:
    """Common pipeline: pattern learning, FAKE tool_use, anticipation, enrichment.

    Shared by proxy mode (process_anthropic_request_lite) and API mode (process_anthropic_request).

    Returns an action dict:
      {"type": "fake", "body": dict}     — respond with fake tool_use
      {"type": "modified", "body": dict} — body was modified (context injected)
      {"type": "passthrough"}            — nothing to do
    """
    prompt = extract_anthropic_prompt(body_dict)
    has_tools = anthropic_has_tools(body_dict)
    model = body_dict.get("model", "")
    triage_prompt = strip_agent_metadata(prompt)
    modified = False

    # ── Agent mode: tools present ──
    if has_tools:
        ctx.anticipation_stats["requests_with_tools"] += 1

        # Pattern learning feedback
        if ctx.pattern_learner:
            try:
                openai_body = anthropic_messages_to_openai(body_dict)
                actual_tools = extract_actual_tool_calls(openai_body.get("messages", []))
                if actual_tools:
                    ctx.pattern_learner.observe_outcome(actual_tools, openai_body.get("messages", []))
            except (AttributeError, TypeError, KeyError):
                pass

        has_tool_result = anthropic_has_tool_result(body_dict)

        # Fake tool_use (high confidence reads)
        if ctx.detector and not has_tool_result:
            try:
                openai_body = anthropic_messages_to_openai(body_dict)
            except (AttributeError, TypeError, KeyError):
                openai_body = {}
            if openai_body:
                prediction = await ctx.detector.predict(openai_body.get("messages", []), openai_body.get("tools"))
                fake_tools = [
                    {"tool": t.tool, "input": dict(t.args)}
                    for t in prediction.tools
                    if t.confidence >= _FAKE_TOOL_CONFIDENCE
                ]
                if fake_tools:
                    ctx.anticipation_stats["predictions_made"] += 1
                    logger.info(
                        "[%s] FAKE tool_use: %s (confidence>=%s)",
                        request_id,
                        [t["tool"] for t in fake_tools],
                        _FAKE_TOOL_CONFIDENCE,
                    )
                    return {"type": "fake", "body": build_fake_anthropic_tool_response(fake_tools, model=model)}

        # NOTE: No local tool pre-execution here. In proxy mode, SmartSplit doesn't have
        # access to the client's filesystem, and even locally it's redundant — the agent
        # executes tools itself in milliseconds. FAKE tool_use above is sufficient.

    # ── Enrichment (workers only, no brain call) ──
    # Skip if in backoff period (previous enrichment caused a 429 from the brain)
    enrichment_allowed = time.time() >= ctx.enrichment_skip_until
    if triage_prompt and enrichment_allowed:
        flat_messages = anthropic_to_flat_messages(body_dict)
        if ctx.enabled:
            decision, enrichment_types = detect(triage_prompt, flat_messages)
            # Filter to only useful enrichments (skip context_summary — redundant when brain has full context)
            enrichment_types = [e for e in enrichment_types if e in _PROXY_USEFUL_ENRICHMENTS]

            # Guard: skip web_search if no search provider AND agent has no search tool
            if "web_search" in enrichment_types:
                has_search_provider = bool(ctx.registry.get_search_providers())
                agent_has_search = anthropic_has_tool_named(body_dict, "web_search", "WebSearch")
                if not has_search_provider and not agent_has_search:
                    enrichment_types = [e for e in enrichment_types if e != "web_search"]
                    logger.info("[%s] Skipped web_search (no provider, no agent tool)", request_id)

            if not enrichment_types:
                decision = TriageDecision.TRANSPARENT
        else:
            decision = TriageDecision.TRANSPARENT

        if decision == TriageDecision.ENRICH:
            try:
                enrichment_results = await enrich_only(ctx, triage_prompt, enrichment_types, messages=flat_messages)
                web_search_failed = "web_search" in enrichment_types and not any(
                    r.type == TaskType.WEB_SEARCH and r.response for r in enrichment_results
                )

                if enrichment_results:
                    worker_context = "\n".join(
                        f"[{r.type.value}]: {r.response}" for r in enrichment_results if r.response
                    )
                    if worker_context:
                        body_dict = inject_anthropic_system_context(body_dict, worker_context)
                        modified = True
                        logger.info("[%s] Enriched with %s worker results", request_id, len(enrichment_results))

                # Cascade: web_search failed but agent has the tool → FAKE tool_use
                if web_search_failed and anthropic_has_tool_named(body_dict, "web_search", "WebSearch"):
                    search_query = getattr(ctx, "_last_search_query", triage_prompt[:200])
                    agent_tool_name = "WebSearch" if anthropic_has_tool_named(body_dict, "WebSearch") else "web_search"
                    logger.info("[%s] Serper failed → FAKE %s(%s)", request_id, agent_tool_name, search_query[:60])
                    fake_tools = [{"tool": agent_tool_name, "input": {"query": search_query}}]
                    return {"type": "fake", "body": build_fake_anthropic_tool_response(fake_tools, model=model)}

            except Exception as exc:
                logger.debug("[%s] Enrichment failed: %s", request_id, type(exc).__name__)
    elif triage_prompt and not enrichment_allowed:
        remaining = int(ctx.enrichment_skip_until - time.time())
        logger.info("[%s] Skipped enrichment (backoff %ss remaining)", request_id, remaining)

    if modified:
        return {"type": "modified", "body": body_dict}

    return {"type": "passthrough"}


async def process_anthropic_request_lite(
    ctx: ProxyContext,
    body_dict: dict,
    request_headers: dict[str, str],
    original_host: str = "api.anthropic.com",
) -> dict:
    """Lightweight pipeline for the TLS proxy — no httpx forwarding.

    Returns an action dict: fake/modified/passthrough.
    The proxy handles the actual relay to the upstream connection.
    """
    return await _run_pipeline(ctx, body_dict, short_id())


async def process_anthropic_request(
    ctx: ProxyContext,
    body_dict: dict,
    request_headers: dict[str, str],
    original_host: str = "api.anthropic.com",
) -> PipelineResult:
    """SmartSplit pipeline for an Anthropic-format request (API mode).

    Unified flow:
      1. Run shared pipeline (FAKE tool_use, anticipation, enrichment)
      2. Forward to brain — strong brain (Claude/GPT) or weak brain (free LLM with planner)
    """
    if not isinstance(body_dict.get("messages"), list):
        return PipelineResult(
            status_code=400,
            body={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "Missing or invalid messages"},
            },
        )

    prompt = extract_anthropic_prompt(body_dict)
    model = body_dict.get("model", "")

    request_start = time.monotonic()
    request_id = short_id()
    logger.info("[%s] Request: %s messages, prompt=%r", request_id, len(body_dict.get("messages", [])), prompt[:80])

    # Determine auth mode (client's own API key vs SmartSplit's brain)
    auth_header = request_headers.get("authorization", "") or request_headers.get("x-api-key", "")
    anthropic_provider = ctx.registry.get("anthropic")
    has_anthropic_key = anthropic_provider is not None and anthropic_provider.api_key
    brain_is_anthropic = ctx.registry.brain_name == "anthropic" or (auth_header and not has_anthropic_key)
    passthrough_headers: dict[str, str] = {}
    if brain_is_anthropic and not has_anthropic_key:
        for key, value in request_headers.items():
            if key.lower() not in ("host", "content-length", "transfer-encoding", "connection"):
                passthrough_headers[key] = value

    # ── 1. Shared pipeline (FAKE tool_use, anticipation, enrichment) ──
    action = await _run_pipeline(ctx, body_dict, request_id)

    if action["type"] == "fake":
        return PipelineResult(body=action["body"])

    if action["type"] == "modified":
        body_dict = action["body"]

    # ── 2. Forward to brain ──
    # Strong brain (Anthropic/OpenAI via passthrough or API key): forward directly
    if brain_is_anthropic or has_anthropic_key:
        try:
            result = await _forward_anthropic(
                ctx, body_dict, original_host, brain_is_anthropic, model, passthrough_headers
            )
            elapsed = int((time.monotonic() - request_start) * 1000)
            logger.info("[%s] Done (forward to brain): %sms", request_id, elapsed)
            return result

        except _AnthropicPassthroughError as e:
            return PipelineResult(status_code=e.status_code, body=e.body)

        except Exception as e:
            logger.warning("[%s] Forward failed: %s: %s", request_id, type(e).__name__, e)
            if passthrough_headers:
                try:
                    result = await _raw_passthrough(ctx, body_dict, original_host, passthrough_headers)
                    return result
                except Exception:
                    pass
            return PipelineResult(
                status_code=502,
                body={"type": "error", "error": {"type": "api_error", "message": "Proxy to brain failed"}},
            )

    # Weak brain (free LLM): use planner to decompose + route
    triage_prompt = strip_agent_metadata(prompt)
    flat_messages = anthropic_to_flat_messages(body_dict)
    if triage_prompt and ctx.enabled:
        decision, enrichment_types = detect(triage_prompt, flat_messages)
    else:
        decision, enrichment_types = TriageDecision.TRANSPARENT, []

    if decision == TriageDecision.ENRICH:
        content, results = await enrich_and_forward(ctx, triage_prompt, enrichment_types, messages=flat_messages)
        if results:
            _log_request(ctx, prompt, ctx.mode, results)
    else:
        content, results = await forward_to_brain(ctx, prompt, messages=flat_messages)

    if content is None:
        return PipelineResult(
            status_code=503,
            body={"type": "error", "error": {"type": "api_error", "message": "All providers unavailable"}},
        )

    prompt_tokens = sum(r.prompt_tokens for r in results)
    completion_tokens = sum(r.completion_tokens for r in results)
    elapsed_ms = int((time.monotonic() - request_start) * 1000)
    logger.info("[%s] Done: %sms, %s subtask(s)", request_id, elapsed_ms, len(results))
    anthropic_resp = _build_anthropic_text_response(content, model, prompt_tokens, completion_tokens)
    return PipelineResult(body=anthropic_resp)


async def _forward_anthropic(
    ctx: ProxyContext,
    body_dict: dict,
    original_host: str,
    brain_is_anthropic: bool,
    requested_model: str,
    passthrough_headers: dict[str, str] | None = None,
) -> PipelineResult:
    """Forward an Anthropic request to the brain. Returns a PipelineResult."""
    if brain_is_anthropic:
        return await _forward_to_anthropic_host(ctx, body_dict, original_host, passthrough_headers or {})
    else:
        return await _forward_to_non_anthropic_brain_result(ctx, body_dict, requested_model)


async def _forward_to_anthropic_host(
    ctx: ProxyContext,
    body_dict: dict,
    host: str,
    passthrough_headers: dict[str, str],
) -> PipelineResult:
    """Forward to an Anthropic-format host with passthrough headers."""
    proxy_body = copy.deepcopy(body_dict)
    if not passthrough_headers:
        proxy_body.pop("stream", None)

    url = f"https://{host}/v1/messages"

    if passthrough_headers:
        logger.info("Passthrough mode: forwarding client headers to %s", host)
        headers = dict(passthrough_headers)
        headers["Content-Type"] = "application/json"
    else:
        provider = ctx.registry.get("anthropic")
        if not provider or not provider.api_key:
            raise Exception("No Anthropic auth available")
        proxy_body["model"] = provider.config.model
        headers = {
            "x-api-key": provider.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    is_streaming = proxy_body.get("stream", False) and passthrough_headers

    if is_streaming:
        req = ctx.http.build_request("POST", url, headers=headers, json=proxy_body)
        response = await ctx.http.send(req, stream=True)

        if response.status_code in (429, 401, 403, 529):
            await response.aclose()
            try:
                error_body = json.loads(await response.aread())
            except Exception:
                error_body = {"type": "error", "error": {"type": "api_error", "message": "rate limited"}}
            raise _AnthropicPassthroughError(response.status_code, error_body)

        if response.status_code >= 400:
            await response.aclose()
            raise Exception(f"{host} returned {response.status_code}")

        logger.info("Proxied request (streaming) to %s successfully", host)
        # Forward all response headers
        resp_headers = {k: v for k, v in response.headers.items() if k.lower() not in _HOP_BY_HOP}
        return PipelineResult(
            streaming_response=response,
            headers=resp_headers,
        )

    else:
        async with asyncio.timeout(_PROVIDER_CALL_TIMEOUT):
            response = await ctx.http.post(url, headers=headers, json=proxy_body)

        if response.status_code in (429, 401, 403, 529):
            try:
                error_body = response.json()
            except Exception:
                error_body = {"type": "error", "error": {"type": "api_error", "message": "rate limited"}}
            raise _AnthropicPassthroughError(response.status_code, error_body)

        response.raise_for_status()
        data = response.json()
        ctx.registry.circuit_breaker.record_success(ctx.registry.brain_name)
        logger.info("Proxied request to %s successfully", host)
        # Forward all response headers
        resp_headers = {k: v for k, v in response.headers.items() if k.lower() not in _HOP_BY_HOP}
        return PipelineResult(body=data, headers=resp_headers)


async def _forward_to_non_anthropic_brain_result(
    ctx: ProxyContext,
    body_dict: dict,
    requested_model: str,
) -> PipelineResult:
    """Convert Anthropic → OpenAI, call non-Anthropic brain, convert back."""
    openai_body = anthropic_messages_to_openai(body_dict)
    openai_body.pop("stream", None)
    openai_response = await ctx.registry.proxy_to_brain(openai_body)
    anthropic_resp = openai_response_to_anthropic(openai_response, requested_model or "smartsplit")
    return PipelineResult(body=anthropic_resp)


async def _raw_passthrough(
    ctx: ProxyContext,
    body_dict: dict,
    host: str,
    passthrough_headers: dict[str, str],
) -> PipelineResult:
    """Emergency fallback: forward request as-is without modification."""
    proxy_body = copy.deepcopy(body_dict)
    proxy_body.pop("stream", None)
    url = f"https://{host}/v1/messages"
    headers = dict(passthrough_headers)
    headers["Content-Type"] = "application/json"
    async with asyncio.timeout(_PROVIDER_CALL_TIMEOUT):
        response = await ctx.http.post(url, headers=headers, json=proxy_body)
        response.raise_for_status()
    return PipelineResult(body=response.json())


# ── Core pipelines ─────────────────────────────────────────────


async def forward_to_brain(
    ctx: ProxyContext,
    prompt: str,
    messages: list[dict[str, str]] | None = None,
) -> tuple[str | None, list[RouteResult]]:
    """TRANSPARENT path — forward directly to the brain, zero intervention."""
    logger.info("TRANSPARENT → brain: %s", ctx.registry.brain_name)
    try:
        content, usage = await ctx.registry.call_brain(prompt, messages=messages)
        result = RouteResult(
            type=TaskType.GENERAL,
            response=content,
            provider=ctx.registry.brain_name,
            termination=TerminationState.COMPLETED,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        _log_request(ctx, prompt, ctx.mode, [result])
        return content, [result]
    except Exception as e:
        logger.error("Brain call failed: %s", type(e).__name__)
        return None, []


# ── HTTP handlers ──────────────────────────────────────────────


_MAX_REQUEST_BYTES = 1_000_000  # 1MB
# Progressive truncation limits — tried in order when context is too long
_TRUNCATION_LIMITS = [400_000, 100_000, 50_000, 20_000]


async def handle_completions(request: Request) -> JSONResponse | StreamingResponse:
    """Handle POST /v1/chat/completions (OpenAI format)."""
    ctx: ProxyContext = request.app.state.ctx

    body = await request.body()
    if len(body) > _MAX_REQUEST_BYTES:
        return JSONResponse(
            {"error": {"message": "Request too large", "type": "invalid_request_error"}},
            status_code=413,
        )
    try:
        body_dict = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(
            {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
            status_code=400,
        )

    try:
        parsed = OpenAIRequest.model_validate(body_dict)
    except (ValueError, TypeError):
        logger.warning("Invalid request format from %s", request.client.host if request.client else "unknown")
        return JSONResponse(
            {"error": {"message": "Invalid request format", "type": "invalid_request_error"}},
            status_code=400,
        )

    prompt = extract_prompt(parsed)
    has_tools = bool(parsed.tools)

    if not prompt:
        return JSONResponse(build_response("Empty prompt.", 0))

    request_start = time.monotonic()
    request_id = short_id()
    logger.info(
        "[%s] New request: %s messages, prompt=%r%s",
        request_id,
        len(parsed.messages),
        prompt[:80],
        f", tools={len(parsed.tools)}" if has_tools else "",
    )
    logger.debug("[%s] Full prompt: %s", request_id, prompt)

    # Pass original messages to preserve conversation context (including tool fields)
    raw_messages = [{"role": m.role, "content": m.content or ""} for m in parsed.messages]

    # Clean prompt for triage: strip agent-injected XML metadata
    triage_prompt = strip_agent_metadata(prompt)
    # If cleaned prompt is empty (message was only metadata), skip triage entirely
    if not triage_prompt:
        decision, enrichment_types = TriageDecision.TRANSPARENT, []
    elif ctx.enabled:
        decision, enrichment_types = detect(triage_prompt, raw_messages)
    else:
        decision, enrichment_types = TriageDecision.TRANSPARENT, []

    # Phase 2: LLM classification for prompts that keywords missed
    if decision == TriageDecision.TRANSPARENT and ctx.enabled and len(triage_prompt.strip()) >= _LLM_DETECT_MIN_CHARS:
        decision, enrichment_types = await detect_with_llm(triage_prompt, ctx.registry)

    ctx.triage_counts[decision] = ctx.triage_counts.get(decision, 0) + 1
    logger.info("[%s] Detector: %s%s", request_id, decision, f" ({enrichment_types})" if enrichment_types else "")

    # ── Mode Agent: triage + enrichment + anticipation + fake tool_use
    if has_tools:
        ctx.anticipation_stats["requests_with_tools"] += 1

        # Observe actual tool calls from previous turns (learning feedback loop)
        if ctx.pattern_learner:
            actual_tools = extract_actual_tool_calls(body_dict.get("messages", []))
            if actual_tools:
                ctx.pattern_learner.observe_outcome(actual_tools, body_dict.get("messages", []))

        # Check if this is a tool_result after a fake we sent → forward to brain
        has_tool_result = any(m.get("role") == "tool" for m in body_dict.get("messages", []))
        if has_tool_result and ctx.pending_fakes.pop(request_id, False):
            logger.info("[%s] Tool results received after fake, forwarding to brain", request_id)

        try:
            # Enrichment: if triage says ENRICH, run workers in parallel (no brain call —
            # the brain is the client's own LLM in agent mode)
            enrichment_task = None
            if decision == TriageDecision.ENRICH:
                enrichment_task = asyncio.create_task(
                    enrich_only(ctx, triage_prompt, enrichment_types, messages=raw_messages)
                )

            # FAKE tool_use: predict tool calls and return immediately (saves 1 LLM round-trip)
            if ctx.detector and not has_tool_result:
                prediction = await ctx.detector.predict(body_dict.get("messages", []), parsed.tools)
                fake_tools = [
                    {"tool": t.tool, "input": dict(t.args)}
                    for t in prediction.tools
                    if t.confidence >= _FAKE_TOOL_CONFIDENCE
                ]
                if fake_tools:
                    ctx.anticipation_stats["predictions_made"] += 1
                    logger.info(
                        "[%s] FAKE tool_use: %s (confidence>=%s)",
                        request_id,
                        [t["tool"] for t in fake_tools],
                        _FAKE_TOOL_CONFIDENCE,
                    )
                    # Cancel enrichment if we're faking (it'll be used when tool_results come back)
                    if enrichment_task:
                        enrichment_task.cancel()
                    fake_response = build_fake_openai_tool_response(fake_tools, model=parsed.model)
                    if parsed.stream:
                        return StreamingResponse(
                            iter_sse(response_to_sse_chunks(fake_response, parsed.model)),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                        )
                    return JSONResponse(fake_response)

            # Inject enrichment results into the request if available
            if enrichment_task:
                enrichment_results = await enrichment_task
                if enrichment_results:
                    from smartsplit.enrichment import _build_enriched_messages

                    body_dict["messages"] = _build_enriched_messages(raw_messages, prompt, enrichment_results)
                    logger.info(
                        "[%s] Enriched agent request with %s worker results", request_id, len(enrichment_results)
                    )

            raw_response = await ctx.registry.proxy_to_brain(body_dict)
            elapsed = int((time.monotonic() - request_start) * 1000)
            logger.info("[%s] Done (agent mode): %sms", request_id, elapsed)

            if parsed.stream:
                return StreamingResponse(
                    iter_sse(response_to_sse_chunks(raw_response, parsed.model)),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

            return JSONResponse(raw_response)
        except Exception as e:
            logger.error("[%s] Agent mode failed: %s", request_id, type(e).__name__)
            return JSONResponse(
                {"error": {"message": "Proxy to brain failed, please retry", "type": "server_error"}},
                status_code=502,
            )

    if decision == TriageDecision.ENRICH:
        content, results = await enrich_and_forward(ctx, prompt, enrichment_types, messages=raw_messages)
        if results:
            _log_request(ctx, prompt, ctx.mode, results)
    else:
        content, results = await forward_to_brain(ctx, prompt, messages=raw_messages)

    # If brain failed, progressively truncate and retry
    if content is None:
        total_chars = sum(len(m["content"]) for m in raw_messages)
        for limit in _TRUNCATION_LIMITS:
            if total_chars <= limit:
                continue
            logger.warning("[%s] Retrying with truncated conversation (%s → %s chars)", request_id, total_chars, limit)
            truncated_messages = _truncate_messages(raw_messages, limit)
            truncated_prompt = next((m["content"] for m in reversed(truncated_messages) if m["role"] == "user"), prompt)
            # Re-detect with truncated context (long history may no longer apply)
            retry_decision, retry_types = detect(truncated_prompt, truncated_messages)
            if retry_decision == TriageDecision.ENRICH:
                content, results = await enrich_and_forward(
                    ctx, truncated_prompt, retry_types, messages=truncated_messages
                )
            else:
                content, results = await forward_to_brain(ctx, truncated_prompt, messages=truncated_messages)
            if content is not None:
                break

    if content is None:
        unhealthy = ctx.registry.circuit_breaker.get_unhealthy()
        msg = "All providers are currently unavailable."
        if unhealthy:
            msg += f" Circuit breaker open for: {', '.join(unhealthy)}."
        msg += " Please check your API keys and try again later."
        logger.error(msg)
        return JSONResponse(
            {"error": {"message": msg, "type": "server_error"}},
            status_code=503,
        )

    tokens = sum(r.estimated_tokens for r in results)
    prompt_tokens = sum(r.prompt_tokens for r in results)
    completion_tokens = sum(r.completion_tokens for r in results)
    elapsed_ms = int((time.monotonic() - request_start) * 1000)
    providers_used = [r.provider for r in results]
    logger.info("[%s] Done: %sms, %s subtask(s), providers=%s", request_id, elapsed_ms, len(results), providers_used)
    logger.debug("[%s] Response preview: %r", request_id, content[:200])

    if parsed.stream:
        return StreamingResponse(
            iter_sse(stream_chunks(content, model=parsed.model)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return JSONResponse(build_response(content, tokens, prompt_tokens, completion_tokens))


async def handle_anthropic_messages(request: Request) -> JSONResponse | StreamingResponse:
    """Handle POST /v1/messages — thin wrapper around process_anthropic_request."""
    ctx: ProxyContext = request.app.state.ctx

    body = await request.body()
    if len(body) > _MAX_REQUEST_BYTES:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error", "message": "Request too large"}},
            status_code=413,
        )
    try:
        body_dict = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error", "message": "Invalid JSON"}},
            status_code=400,
        )

    result = await process_anthropic_request(ctx, body_dict, dict(request.headers))
    return _pipeline_result_to_starlette(result, bool(body_dict.get("stream", False)), body_dict.get("model", ""))


def _pipeline_result_to_starlette(
    result: PipelineResult,
    wants_stream: bool,
    model: str,
) -> JSONResponse | StreamingResponse:
    """Convert a PipelineResult to a Starlette response."""
    if result.is_streaming:
        stream_resp = result.streaming_response

        async def _stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in stream_resp.aiter_bytes():
                    yield chunk
            finally:
                await stream_resp.aclose()

        return StreamingResponse(
            _stream(),
            media_type=result.headers.get("content-type", "text/event-stream"),
            status_code=result.status_code,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    body = result.body or {}

    if wants_stream and result.status_code == 200:
        return StreamingResponse(
            iter_sse(anthropic_response_to_sse(body)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return JSONResponse(body, status_code=result.status_code)


# ── Anthropic helpers ─────────────────────────────────────────


def _build_anthropic_text_response(
    content: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict:
    """Build an Anthropic Messages API response with a single text block."""
    return {
        "id": "msg_" + _uuid.uuid4().hex[:24],
        "type": "message",
        "role": "assistant",
        "model": model or "smartsplit",
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        },
    }


async def handle_health(request: Request) -> JSONResponse:
    """GET /health."""
    ctx: ProxyContext = request.app.state.ctx
    return JSONResponse(
        {
            "status": "ok",
            "enabled": ctx.enabled,
            "mode": ctx.mode.value,
            "brain": ctx.registry.brain_name,
            "providers": len(ctx.config.providers),
        }
    )


async def handle_savings(request: Request) -> JSONResponse:
    """GET /savings."""
    ctx: ProxyContext = request.app.state.ctx
    return JSONResponse(ctx.quota.get_savings_report().model_dump())


async def handle_metrics(request: Request) -> JSONResponse:
    """GET /metrics."""
    ctx: ProxyContext = request.app.state.ctx
    savings = ctx.quota.get_savings_report()
    total = sum(ctx.triage_counts.values())

    return JSONResponse(
        {
            "requests": {
                "total": total,
                "transparent": ctx.triage_counts.get(TriageDecision.TRANSPARENT, 0),
                "enrich": ctx.triage_counts.get(TriageDecision.ENRICH, 0),
            },
            "savings": {
                "tokens_saved": savings.estimated_tokens_saved,
                "cost_saved_usd": savings.estimated_cost_saved_usd,
                "free_percentage": savings.free_percentage,
            },
            "providers": savings.providers_usage,
            "cache": {
                "hits": ctx.planner.cache.hits,
                "misses": ctx.planner.cache.misses,
                "hit_rate": round(
                    ctx.planner.cache.hits / max(ctx.planner.cache.hits + ctx.planner.cache.misses, 1) * 100,
                    1,
                ),
            },
            "circuit_breaker": {
                "unhealthy_providers": ctx.registry.circuit_breaker.get_unhealthy(),
            },
            "learning": ctx.bandit.get_stats(),
            "anticipation": {
                **ctx.anticipation_stats,
                "pattern_learner": ctx.pattern_learner.get_stats() if ctx.pattern_learner else {},
            },
            "mode": ctx.mode.value,
            "enabled": ctx.enabled,
        }
    )


# ── Request logging ───────────────────────────────────────────


def _log_request(
    ctx: ProxyContext,
    prompt: str,
    mode: Mode,
    results: list[RouteResult],
    domains: list[str] | None = None,
) -> None:
    """Record a request in the in-memory log."""

    steps = []
    free_calls = paid_calls = 0

    for r in results:
        pconfig = ctx.config.providers.get(r.provider)
        is_paid = pconfig.type == ProviderType.PAID if pconfig else False
        steps.append(
            RoutingStep(
                task_type=r.type,
                provider=r.provider,
                score=round(r.score, 3),
                is_paid=is_paid,
                termination=r.termination,
                escalations=r.escalations,
                estimated_tokens=r.estimated_tokens,
            )
        )
        if is_paid:
            paid_calls += 1
        else:
            free_calls += 1

    ctx.request_logs.append(
        RequestLog(
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            prompt_preview=prompt[:100],
            mode=mode,
            domains_detected=domains or [],
            subtask_count=len(results),
            steps=steps,
            free_calls=free_calls,
            paid_calls=paid_calls,
            tokens_saved=sum(s.estimated_tokens for s in steps if not s.is_paid),
        )
    )


# ── App factory ────────────────────────────────────────────────


def create_app(
    config: SmartSplitConfig | None = None,
    mode: str = "balanced",
) -> Starlette:
    """Create the Starlette ASGI app."""

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        cfg = config if config is not None else load_config()

        http = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
        quota = QuotaTracker(provider_configs=cfg.providers)
        registry = ProviderRegistry(cfg.providers, http, free_llm_priority=cfg.free_llm_priority, brain_name=cfg.brain)
        planner = Planner(registry)
        bandit = BanditScorer()
        router = Router(registry, quota, cfg, bandit=bandit)

        # V3 anticipation components
        pattern_learner = ToolPatternLearner(project_dir=".")
        detector = IntentionDetector(registry, pattern_learner=pattern_learner)
        anticipator = ToolAnticipator(registry, working_dir=".")

        app.state.ctx = ProxyContext(
            config=cfg,
            registry=registry,
            planner=planner,
            router=router,
            quota=quota,
            bandit=bandit,
            http=http,
            mode=Mode(mode),
            detector=detector,
            anticipator=anticipator,
            pattern_learner=pattern_learner,
        )
        logger.info("SmartSplit started (mode=%s, brain=%s, providers=%s)", mode, cfg.brain, len(cfg.providers))

        # Log override status
        if cfg.overrides:
            for task_type, provider_name in cfg.overrides.items():
                pconfig = cfg.providers.get(provider_name)
                if pconfig and pconfig.enabled and pconfig.api_key:
                    logger.info("Override: %s → %s ✓", task_type, provider_name)
                else:
                    logger.warning("Override: %s → %s ✗ (not configured)", task_type, provider_name)

        try:
            yield
        finally:
            quota.flush()
            bandit.flush()
            pattern_learner.flush()
            await http.aclose()
            logger.info("SmartSplit stopped")

    routes = [
        Route("/v1/chat/completions", handle_completions, methods=["POST"]),
        Route("/v1/messages", handle_anthropic_messages, methods=["POST"]),
        Route("/health", handle_health, methods=["GET"]),
        Route("/savings", handle_savings, methods=["GET"]),
        Route("/metrics", handle_metrics, methods=["GET"]),
    ]

    return Starlette(routes=routes, lifespan=lifespan)
