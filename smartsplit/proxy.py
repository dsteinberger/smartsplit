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
from smartsplit.models import (
    Mode,
    ProviderType,
    RequestLog,
    RouteResult,
    RoutingStep,
    Subtask,
    TaskType,
)
from smartsplit.planner import Planner, detect_domains
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


def _get_last_user_text(messages: list[dict]) -> str:
    """Extract the text of the last user message."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts)
    return ""


def triage(request_body: dict) -> str:
    """Classify a request into ENRICH or RESPOND.

    ENRICH when the prompt touches web_search, factual, or multi-domain topics
    that benefit from external data.

    RESPOND for everything else — route to the best free LLM directly.
    """
    messages = request_body.get("messages", [])
    if not messages:
        return TriageDecision.RESPOND

    last_user_text = _get_last_user_text(messages)
    if not last_user_text:
        return TriageDecision.RESPOND

    domains = detect_domains(last_user_text)
    domain_names = [d for d, _ in domains]

    # Multi-domain or has enrichable component → ENRICH
    if any(d in ("web_search", "factual", "summarize") for d in domain_names):
        return TriageDecision.ENRICH

    # Multi-domain prompt benefits from enrichment
    if len(domain_names) > 1:
        return TriageDecision.ENRICH

    return TriageDecision.RESPOND


# ── Server context ────────────────────────────────────────────


@dataclass
class ProxyContext:
    config: SmartSplitConfig
    registry: ProviderRegistry
    planner: Planner
    router: Router
    quota: QuotaTracker
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
) -> tuple[str, list[RouteResult]]:
    """Route to the best free LLM for each subtask. Core pipeline."""
    domains = [d for d, _ in detect_domains(prompt)]
    subtasks = await ctx.planner.decompose(prompt, mode=mode)

    logger.info(f"RESPOND: {len(subtasks)} subtask(s) via free providers")

    results = await asyncio.gather(*(ctx.router.route(st, mode) for st in subtasks))
    results = [r for r in results if r.response]

    if not results:
        return "No provider could handle this request. Please check your API keys.", []

    _log_request(ctx, prompt, mode, results, domains)

    if len(results) == 1:
        return results[0].response, results

    synthesis = await ctx.planner.synthesize(prompt, results)
    return synthesis, results


async def enrich_and_respond(
    ctx: ProxyContext,
    prompt: str,
    mode: Mode,
) -> tuple[str, list[RouteResult]]:
    """Web search + summaries first, then route to best free LLM."""
    domains = [d for d, _ in detect_domains(prompt)]
    subtasks = await ctx.planner.decompose(prompt, mode=mode)

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
                )
                for st in main_tasks
            ]
        main_results = await asyncio.gather(*(ctx.router.route(st, mode) for st in main_tasks))
        main_results = [r for r in main_results if r.response]
    else:
        main_results = []

    all_results = enrichment_results + main_results

    if not all_results:
        return "No provider could handle this request. Please check your API keys.", []

    _log_request(ctx, prompt, mode, all_results, domains)

    if len(all_results) == 1:
        return all_results[0].response, all_results

    synthesis = await ctx.planner.synthesize(prompt, all_results)
    return synthesis, all_results


# ── HTTP handlers ──────────────────────────────────────────────


_MAX_REQUEST_BYTES = 1_000_000  # 1MB


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
    except Exception:
        logger.warning(f"Invalid request format from {request.client}")
        return JSONResponse(
            {"error": {"message": "Invalid request format", "type": "invalid_request_error"}},
            status_code=400,
        )

    prompt = extract_prompt(parsed)

    if not prompt:
        return JSONResponse(build_response("Empty prompt.", 0))

    decision = triage(body_dict) if ctx.enabled else TriageDecision.RESPOND
    ctx.triage_counts[decision] = ctx.triage_counts.get(decision, 0) + 1
    logger.info(f"Triage: {decision}")

    if decision == TriageDecision.ENRICH:
        content, results = await enrich_and_respond(ctx, prompt, ctx.mode)
    else:
        content, results = await respond_directly(ctx, prompt, ctx.mode)

    tokens = sum(r.estimated_tokens for r in results)

    if parsed.stream:
        return StreamingResponse(
            _iter_sse(stream_chunks(content, model=parsed.model)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return JSONResponse(build_response(content, tokens))


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

        http = httpx.AsyncClient(timeout=30.0)
        quota = QuotaTracker(provider_configs=cfg.providers)
        registry = ProviderRegistry(cfg.providers, http)
        planner = Planner(registry)
        router = Router(registry, quota, cfg)

        app.state.ctx = ProxyContext(
            config=cfg,
            registry=registry,
            planner=planner,
            router=router,
            quota=quota,
            http=http,
            mode=Mode(mode),
        )
        logger.info(f"SmartSplit started (mode={mode}, providers={len(cfg.providers)})")

        try:
            yield
        finally:
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
