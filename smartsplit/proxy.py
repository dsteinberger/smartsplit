"""SmartSplit — Free multi-LLM backend with intelligent routing.

An OpenAI-compatible endpoint that routes every request to the best free LLM
for the task. Works as a drop-in backend for Continue, Cline, Aider, or any
OpenAI-compatible client.

Usage:
    smartsplit --port 8420
    # Then point your client at http://localhost:8420/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from smartsplit.config import SmartSplitConfig, load_config
from smartsplit.formats import (
    OpenAIRequest,
    build_response,
    extract_prompt,
    stream_chunks,
)
from smartsplit.learning import BanditScorer
from smartsplit.models import (
    Mode,
    ProviderType,
    RequestLog,
    RouteResult,
    RoutingStep,
    Subtask,
    TaskType,
    TerminationState,
    short_id,
)
from smartsplit.planner import Planner
from smartsplit.providers.registry import ProviderRegistry
from smartsplit.quota import QuotaTracker
from smartsplit.router import Router

logger = logging.getLogger("smartsplit.proxy")

_MAX_LOG_ENTRIES = 50


# ── Triage: 2 automatic modes ─────────────────────────────────
#
# Every request is classified into one of two modes:
#   RESPOND  — route to the best free LLM for the task (default)
#   ENRICH   — web search + summaries first, then route to best free LLM


class TriageDecision(StrEnum):
    ENRICH = "enrich"  # Add free web context, then respond
    RESPOND = "respond"  # Route directly to best free LLM


async def triage(prompt: str, planner: Planner) -> tuple[TriageDecision, list[str]]:
    """Classify a prompt into ENRICH or RESPOND using LLM classification.

    Uses the planner's LLM-based domain classifier (with keyword fallback)
    for accurate, language-agnostic detection. The classification result
    is reused by decompose() via the planner's internal cache.

    ENRICH when the prompt needs external data (web_search, factual).
    RESPOND for everything else — route to the best LLM directly.
    """
    if not prompt:
        return TriageDecision.RESPOND, []

    domain_names = await planner.classify_domains(prompt)

    # ENRICH only when external data is genuinely needed
    if any(d in ("web_search", "factual") for d in domain_names):
        return TriageDecision.ENRICH, domain_names

    return TriageDecision.RESPOND, domain_names


# ── Server context ────────────────────────────────────────────


@dataclass
class ProxyContext:
    config: SmartSplitConfig
    registry: ProviderRegistry
    planner: Planner
    router: Router
    quota: QuotaTracker
    bandit: BanditScorer
    http: httpx.AsyncClient
    request_logs: deque[RequestLog] = field(default_factory=lambda: deque(maxlen=_MAX_LOG_ENTRIES))
    mode: Mode = Mode.BALANCED
    enabled: bool = True
    triage_counts: dict[str, int] = field(
        default_factory=lambda: {
            TriageDecision.ENRICH: 0,
            TriageDecision.RESPOND: 0,
        }
    )


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
        kept = [{"role": last["role"], "content": last["content"][: max(budget, 500)]}]

    return system + kept


# ── Core pipelines ─────────────────────────────────────────────

# Subtask types safe for web enrichment
_ENRICHABLE_TYPES = {
    TaskType.WEB_SEARCH,
    TaskType.SUMMARIZE,
    TaskType.FACTUAL,
    TaskType.TRANSLATION,
}


async def respond_directly(
    ctx: ProxyContext,
    prompt: str,
    mode: Mode,
    domains: list[str] | None = None,
    messages: list[dict[str, str]] | None = None,
) -> tuple[str | None, list[RouteResult]]:
    """Route to the best free LLM for each subtask. Core pipeline."""
    subtasks = await ctx.planner.decompose(prompt, mode=mode, messages=messages, domains=domains)

    logger.info(f"RESPOND: {len(subtasks)} subtask(s) via free providers")

    results = await asyncio.gather(*(ctx.router.route(st, mode) for st in subtasks))
    results = [r for r in results if r.response]

    _FAILED_STATES = {TerminationState.ALL_FAILED, TerminationState.NO_PROVIDER}
    all_failed = all(r.termination in _FAILED_STATES for r in results) if results else True
    if not results or all_failed:
        return None, []

    _log_request(ctx, prompt, mode, results, domains)

    if len(results) == 1:
        return results[0].response, results

    synthesis = await ctx.planner.synthesize(prompt, results)
    return synthesis, results


async def enrich_and_respond(
    ctx: ProxyContext,
    prompt: str,
    mode: Mode,
    domains: list[str] | None = None,
    messages: list[dict[str, str]] | None = None,
) -> tuple[str | None, list[RouteResult]]:
    """Web search + summaries first, then route to best free LLM."""
    subtasks = await ctx.planner.decompose(prompt, mode=mode, messages=messages, domains=domains)

    logger.info(f"ENRICH: {len(subtasks)} subtask(s)")

    # Separate enrichable (web search, summaries) from the rest
    enrichable = [st for st in subtasks if st.type in _ENRICHABLE_TYPES]
    main_tasks = [st for st in subtasks if st.type not in _ENRICHABLE_TYPES]

    # Run enrichable subtasks first
    enrichment_results = []
    if enrichable:
        raw = await asyncio.gather(*(ctx.router.route(st, mode) for st in enrichable))
        enrichment_results = [r for r in raw if r.response]

    # Build enrichment context for main tasks
    enrichment_context = ""
    if enrichment_results:
        parts = [f"[{r.type.value}] {r.response}" for r in enrichment_results]
        enrichment_context = "\n\n".join(parts)

    # Run main tasks with enrichment injected
    if main_tasks:
        if enrichment_context:
            main_tasks = [
                Subtask(
                    type=st.type,
                    content=f"{st.content}\n\nContext:\n{enrichment_context}",
                    complexity=st.complexity,
                    depends_on=st.depends_on,
                    messages=st.messages,
                )
                for st in main_tasks
            ]
        main_results = await asyncio.gather(*(ctx.router.route(st, mode) for st in main_tasks))
        main_results = [r for r in main_results if r.response]
    else:
        main_results = []

    all_results = enrichment_results + main_results

    _FAILED_STATES = {TerminationState.ALL_FAILED, TerminationState.NO_PROVIDER}
    all_failed = all(r.termination in _FAILED_STATES for r in all_results) if all_results else True
    if not all_results or all_failed:
        return None, []

    _log_request(ctx, prompt, mode, all_results, domains)

    if len(all_results) == 1:
        return all_results[0].response, all_results

    synthesis = await ctx.planner.synthesize(prompt, all_results)
    return synthesis, all_results


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
        logger.warning(f"Invalid request format from {request.client.host if request.client else 'unknown'}")
        return JSONResponse(
            {"error": {"message": "Invalid request format", "type": "invalid_request_error"}},
            status_code=400,
        )

    prompt = extract_prompt(parsed)

    if not prompt:
        return JSONResponse(build_response("Empty prompt.", 0))

    request_start = time.monotonic()
    request_id = short_id()
    logger.info(f"[{request_id}] New request: {len(parsed.messages)} messages, prompt={prompt[:80]!r}")
    logger.debug(f"[{request_id}] Full prompt: {prompt}")

    decision, domains = await triage(prompt, ctx.planner) if ctx.enabled else (TriageDecision.RESPOND, [])
    ctx.triage_counts[decision] = ctx.triage_counts.get(decision, 0) + 1
    logger.info(f"[{request_id}] Triage: {decision} (domains={domains})")

    # Pass original messages to preserve conversation context
    raw_messages = [{"role": m.role, "content": m.content} for m in parsed.messages]

    if decision == TriageDecision.ENRICH:
        content, results = await enrich_and_respond(ctx, prompt, ctx.mode, domains=domains, messages=raw_messages)
    else:
        content, results = await respond_directly(ctx, prompt, ctx.mode, domains=domains, messages=raw_messages)

    # If all providers failed, progressively truncate and retry
    if content is None:
        total_chars = sum(len(m["content"]) for m in raw_messages)
        for limit in _TRUNCATION_LIMITS:
            if total_chars <= limit:
                continue  # conversation already fits, truncation won't help
            logger.warning(f"[{request_id}] Retrying with truncated conversation ({total_chars} → {limit} chars)")
            truncated_messages = _truncate_messages(raw_messages, limit)
            truncated_prompt = next((m["content"] for m in reversed(truncated_messages) if m["role"] == "user"), prompt)
            if decision == TriageDecision.ENRICH:
                content, results = await enrich_and_respond(
                    ctx, truncated_prompt, ctx.mode, domains=domains, messages=truncated_messages
                )
            else:
                content, results = await respond_directly(
                    ctx, truncated_prompt, ctx.mode, domains=domains, messages=truncated_messages
                )
            if content is not None:
                break  # success — stop truncating

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
    logger.info(f"[{request_id}] Done: {elapsed_ms}ms, {len(results)} subtask(s), providers={providers_used}")
    logger.debug(f"[{request_id}] Response preview: {content[:200]!r}")

    if parsed.stream:
        return StreamingResponse(
            _iter_sse(stream_chunks(content, model=parsed.model)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return JSONResponse(build_response(content, tokens, prompt_tokens, completion_tokens))


async def handle_health(request: Request) -> JSONResponse:
    """GET /health."""
    ctx: ProxyContext = request.app.state.ctx
    return JSONResponse(
        {
            "status": "ok",
            "enabled": ctx.enabled,
            "mode": ctx.mode.value,
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
                "enrich": ctx.triage_counts.get(TriageDecision.ENRICH, 0),
                "respond": ctx.triage_counts.get(TriageDecision.RESPOND, 0),
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
            "mode": ctx.mode.value,
            "enabled": ctx.enabled,
        }
    )


# ── Helpers ────────────────────────────────────────────────────


async def _iter_sse(chunks: list[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


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
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
        quota = QuotaTracker(provider_configs=cfg.providers)
        registry = ProviderRegistry(cfg.providers, http, free_llm_priority=cfg.free_llm_priority)
        planner = Planner(registry)
        bandit = BanditScorer()
        router = Router(registry, quota, cfg, bandit=bandit)

        app.state.ctx = ProxyContext(
            config=cfg,
            registry=registry,
            planner=planner,
            router=router,
            quota=quota,
            bandit=bandit,
            http=http,
            mode=Mode(mode),
        )
        logger.info(f"SmartSplit started (mode={mode}, providers={len(cfg.providers)})")

        try:
            yield
        finally:
            quota.flush()
            bandit.flush()
            await http.aclose()
            logger.info("SmartSplit stopped")

    routes = [
        Route("/v1/chat/completions", handle_completions, methods=["POST"]),
        Route("/health", handle_health, methods=["GET"]),
        Route("/savings", handle_savings, methods=["GET"]),
        Route("/metrics", handle_metrics, methods=["GET"]),
    ]

    return Starlette(routes=routes, lifespan=lifespan)


# ── CLI ────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SmartSplit — Free multi-LLM backend with intelligent routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  smartsplit                          # Start on port 8420
  smartsplit --port 3456 --mode economy

Then in your client:
  Continue: apiBase: http://localhost:8420/v1
  Cline:    cline auth -b http://localhost:8420/v1
  Aider:    aider --openai-api-base http://localhost:8420/v1
        """,
    )
    parser.add_argument("--port", type=int, default=8420, help="Port (default: 8420)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument(
        "--mode", type=str, default="balanced", choices=["economy", "balanced", "quality"], help="Routing mode"
    )
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import uvicorn

    app = create_app(mode=args.mode)

    print("\n  SmartSplit — Free multi-LLM backend")
    print(f"  http://{args.host}:{args.port}/v1")
    print(f"  Mode: {args.mode}")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
