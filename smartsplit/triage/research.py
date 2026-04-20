"""Mini research agent — PLAN → SEARCH → READ → GAP pipeline with strict time budget.

Replaces the one-shot ``web_search`` enrichment with a staged pipeline:
    1. PLAN  — decompose user prompt into 2-3 targeted search queries
    2. SEARCH — Serper in parallel on each query
    3. READ  — worker LLM reads snippets, extracts sourced findings + gaps
    4. GAP   — (optional) single follow-up search if critical gaps detected

Each step has a fallback that degrades gracefully to the previous state:
- PLAN fail   → [prompt[:200]] as single raw query
- READ fail   → raw snippets string (same behavior as before this refactor)
- GAP skipped → findings returned unchanged

The whole pipeline is bounded by a global budget (default 7s). Before each
step we check how much time is left and either proceed or fall back.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from smartsplit.json_utils import extract_json
from smartsplit.models import ResearchFinding, ResearchReport
from smartsplit.tools.anticipation import extract_project_context
from smartsplit.triage.research_prompts import (
    GAP_PROMPT,
    PLAN_PROMPT,
    SYNTHESIZE_PROMPT,
)

if TYPE_CHECKING:
    from smartsplit.proxy.pipeline import ProxyContext

logger = logging.getLogger("smartsplit.research")

# Step minimum budgets (seconds). Below these we skip the step and degrade.
_MIN_PLAN = 1.0
_MIN_SEARCH = 2.0
_MIN_READ = 2.0
_MIN_GAP = 2.0

# Soft limits
_MAX_QUERIES = 3
_MAX_FINDINGS = 6
_MAX_QUERY_LEN = 120
_MAX_PROMPT_CHARS_FOR_PLAN = 500


# ── Budget ───────────────────────────────────────────────────


class Budget:
    """Deadline-based time budget for multi-step pipelines.

    Usage::

        b = Budget(7.0)
        if b.has_at_least(2.0):
            await asyncio.wait_for(step(), timeout=b.remaining())
    """

    def __init__(self, total_seconds: float) -> None:
        self.total = total_seconds
        self.deadline = time.monotonic() + total_seconds

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def has_at_least(self, seconds: float) -> bool:
        return self.remaining() >= seconds

    def child(self, fraction: float) -> float:
        """Return ``fraction`` of the remaining budget (bounded to remaining)."""
        return max(0.0, self.remaining() * fraction)


# ── Step 1: PLAN ─────────────────────────────────────────────


def _fallback_queries(prompt: str) -> list[str]:
    """Single raw query derived from the prompt — used when PLAN fails."""
    return [prompt[:200].strip()] if prompt.strip() else []


async def plan_queries(
    ctx: ProxyContext,
    prompt: str,
    messages: list[dict] | None,
    *,
    budget: Budget,
) -> list[str]:
    """Decompose ``prompt`` into 1-3 targeted search queries via worker LLM.

    Falls back to a single raw query on timeout, parse error, or insufficient budget.
    Never raises — always returns at least one usable query (unless prompt is empty).
    """
    if not budget.has_at_least(_MIN_PLAN):
        logger.info("PLAN skipped: budget too tight (%.2fs left)", budget.remaining())
        return _fallback_queries(prompt)

    context = extract_project_context(messages or [])
    full_prompt = PLAN_PROMPT.replace("{context}", context).replace("{prompt}", prompt[:_MAX_PROMPT_CHARS_FOR_PLAN])

    try:
        timeout = min(budget.remaining(), max(_MIN_PLAN, budget.child(0.3)))
        raw = await asyncio.wait_for(
            ctx.registry.call_worker_llm(full_prompt, prefer="cerebras"),
            timeout=timeout,
        )
    except TimeoutError:
        logger.info("PLAN timed out, falling back to raw prompt query")
        return _fallback_queries(prompt)
    except Exception as e:
        logger.info("PLAN failed (%s), falling back to raw prompt query", type(e).__name__)
        return _fallback_queries(prompt)

    try:
        parsed = json.loads(extract_json(raw))
    except (ValueError, json.JSONDecodeError):
        logger.info("PLAN returned non-JSON, falling back to raw prompt query")
        return _fallback_queries(prompt)

    if not isinstance(parsed, list) or not parsed:
        return _fallback_queries(prompt)

    queries = []
    for q in parsed[:_MAX_QUERIES]:
        if not isinstance(q, str):
            continue
        cleaned = q.strip()[:_MAX_QUERY_LEN]
        if cleaned:
            queries.append(cleaned)

    if not queries:
        return _fallback_queries(prompt)

    logger.info("PLAN produced %d queries: %s", len(queries), queries)
    return queries


# ── Step 2: SEARCH ───────────────────────────────────────────


async def _search_one(ctx: ProxyContext, query: str, timeout: float) -> tuple[str, str] | None:
    """Run one search, return (query, raw_snippets) or None on failure."""
    providers = ctx.registry.get_search_providers()
    if not providers:
        return None
    # Use the first available search provider (Serper preferred, Tavily as fallback)
    name, provider = next(iter(providers.items()))
    try:
        snippets = await asyncio.wait_for(provider.search(query), timeout=timeout)
    except TimeoutError:
        logger.info("SEARCH %s timed out on %r", name, query[:60])
        return None
    except Exception as e:
        logger.info("SEARCH %s failed on %r: %s", name, query[:60], type(e).__name__)
        return None
    if not snippets or not snippets.strip():
        return None
    return query, snippets


async def search_parallel(
    ctx: ProxyContext,
    queries: list[str],
    *,
    budget: Budget,
) -> list[tuple[str, str]]:
    """Run all ``queries`` in parallel against the first available search provider.

    Drops queries that fail or time out; returns whatever succeeded.
    Returns an empty list if no search provider is configured or budget is tight.
    """
    if not queries:
        return []
    if not budget.has_at_least(_MIN_SEARCH):
        logger.info("SEARCH skipped: budget too tight (%.2fs left)", budget.remaining())
        return []
    if not ctx.registry.get_search_providers():
        logger.info("SEARCH skipped: no search provider configured")
        return []

    per_query_timeout = min(budget.remaining(), max(_MIN_SEARCH, budget.child(0.5)))
    raw = await asyncio.gather(
        *(_search_one(ctx, q, per_query_timeout) for q in queries),
        return_exceptions=True,
    )
    results: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, tuple):
            results.append(item)
    logger.info("SEARCH completed: %d/%d succeeded", len(results), len(queries))
    return results


# ── Step 3: READ + SYNTHESIZE ────────────────────────────────


def _format_search_results(search_results: list[tuple[str, str]]) -> str:
    """Flatten [(query, snippets), ...] into a prompt-friendly block."""
    parts = []
    for query, snippets in search_results:
        parts.append(f"### Query: {query}\n{snippets}")
    return "\n\n".join(parts)


def _raw_snippets_fallback(search_results: list[tuple[str, str]], queries: list[str]) -> ResearchReport:
    """Build a report with no findings but containing the queries — caller will use raw snippets."""
    return ResearchReport(
        findings=[],
        gaps=["synthesis failed — using raw snippets"],
        queries_used=queries,
    )


def _parse_findings(parsed: dict) -> list[ResearchFinding]:
    """Extract valid findings from the LLM's JSON output. Drop anything malformed."""
    raw = parsed.get("findings", [])
    if not isinstance(raw, list):
        return []

    findings: list[ResearchFinding] = []
    for item in raw[:_MAX_FINDINGS]:
        if not isinstance(item, dict):
            continue
        fact = item.get("fact")
        url = item.get("source_url")
        if not isinstance(fact, str) or not isinstance(url, str):
            continue
        if not fact.strip() or not url.strip():
            continue
        confidence = item.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
        findings.append(ResearchFinding(fact=fact.strip(), source_url=url.strip(), confidence=confidence))
    return findings


def _parse_gaps(parsed: dict) -> list[str]:
    raw = parsed.get("gaps", [])
    if not isinstance(raw, list):
        return []
    return [g.strip() for g in raw if isinstance(g, str) and g.strip()]


async def read_and_synthesize(
    ctx: ProxyContext,
    prompt: str,
    search_results: list[tuple[str, str]],
    *,
    budget: Budget,
) -> ResearchReport:
    """Read snippets via worker LLM, return a structured ResearchReport.

    Falls back to an empty-findings report (caller will reinject raw snippets)
    on timeout, invalid JSON, or insufficient budget.
    """
    queries = [q for q, _ in search_results]

    if not search_results:
        return ResearchReport(findings=[], gaps=[], queries_used=queries)

    if not budget.has_at_least(_MIN_READ):
        logger.info("READ skipped: budget too tight (%.2fs left)", budget.remaining())
        return _raw_snippets_fallback(search_results, queries)

    results_block = _format_search_results(search_results)
    # Keep the prompt bounded — larger context tiers can take more, but we
    # rely on call_worker_llm's own truncation for that.
    full_prompt = SYNTHESIZE_PROMPT.replace("{prompt}", prompt[:_MAX_PROMPT_CHARS_FOR_PLAN]).replace(
        "{search_results}", results_block
    )

    try:
        timeout = min(budget.remaining(), max(_MIN_READ, budget.child(0.7)))
        raw = await asyncio.wait_for(
            # Prefer gemini — larger context tier, better at structured JSON than cerebras on long inputs
            ctx.registry.call_worker_llm(full_prompt, prefer="gemini"),
            timeout=timeout,
        )
    except TimeoutError:
        logger.info("READ timed out, falling back to raw snippets")
        return _raw_snippets_fallback(search_results, queries)
    except Exception as e:
        logger.info("READ failed (%s), falling back to raw snippets", type(e).__name__)
        return _raw_snippets_fallback(search_results, queries)

    try:
        parsed = json.loads(extract_json(raw))
    except (ValueError, json.JSONDecodeError):
        logger.info("READ returned non-JSON, falling back to raw snippets")
        return _raw_snippets_fallback(search_results, queries)

    if not isinstance(parsed, dict):
        return _raw_snippets_fallback(search_results, queries)

    findings = _parse_findings(parsed)
    gaps = _parse_gaps(parsed)

    if not findings:
        # LLM returned valid JSON but no sourced findings — treat as degraded
        logger.info("READ returned 0 findings, falling back to raw snippets")
        return _raw_snippets_fallback(search_results, queries)

    logger.info("READ synthesized %d findings, %d gaps", len(findings), len(gaps))
    return ResearchReport(findings=findings, gaps=gaps, queries_used=queries)


# ── Step 4: GAP FILL ─────────────────────────────────────────


async def _plan_gap_query(
    ctx: ProxyContext,
    prompt: str,
    gaps: list[str],
    *,
    timeout: float,
) -> str | None:
    """Ask a worker LLM for a single follow-up query to fill the reported gaps."""
    full_prompt = GAP_PROMPT.replace("{gaps}", "\n".join(f"- {g}" for g in gaps)).replace(
        "{prompt}", prompt[:_MAX_PROMPT_CHARS_FOR_PLAN]
    )
    try:
        raw = await asyncio.wait_for(
            ctx.registry.call_worker_llm(full_prompt, prefer="cerebras"),
            timeout=timeout,
        )
    except Exception as e:
        logger.info("GAP plan failed (%s)", type(e).__name__)
        return None

    cleaned = extract_json(raw).strip()
    # Accept either a bare quoted string or an array
    try:
        parsed = json.loads(cleaned)
    except (ValueError, json.JSONDecodeError):
        return None

    if isinstance(parsed, str) and parsed.strip():
        return parsed.strip()[:_MAX_QUERY_LEN]
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
        return parsed[0].strip()[:_MAX_QUERY_LEN] or None
    return None


async def gap_fill(
    ctx: ProxyContext,
    prompt: str,
    report: ResearchReport,
    *,
    budget: Budget,
) -> ResearchReport:
    """Run ONE follow-up search+synthesis to fill the most critical gap.

    Skipped if:
    - report has no gaps
    - budget < _MIN_GAP seconds
    - follow-up query planning or search fails

    Returns the (possibly enriched) report. Never raises.
    """
    if not report.gaps:
        return report
    # If READ already degraded (synthesis failed), gap-fill makes no sense
    if report.gaps and report.gaps[0].startswith("synthesis failed"):
        return report
    if not budget.has_at_least(_MIN_GAP):
        logger.info("GAP skipped: budget too tight (%.2fs left)", budget.remaining())
        return report

    # Plan one query — use a small slice of the budget for this
    plan_timeout = min(budget.remaining(), max(1.0, budget.child(0.2)))
    query = await _plan_gap_query(ctx, prompt, report.gaps, timeout=plan_timeout)
    if not query:
        return report

    # Search
    if not budget.has_at_least(_MIN_GAP / 2):
        return report
    search_timeout = min(budget.remaining(), max(1.0, budget.child(0.4)))
    search_hit = await _search_one(ctx, query, search_timeout)
    if search_hit is None:
        return report

    # Synthesize — reuse read_and_synthesize on the extra result, then merge
    if not budget.has_at_least(_MIN_READ):
        return report
    extra = await read_and_synthesize(ctx, prompt, [search_hit], budget=budget)

    merged_findings = list(report.findings) + list(extra.findings)
    # The follow-up "fills" the gaps that motivated it, so drop them
    remaining_gaps = list(extra.gaps) if extra.findings else list(report.gaps)
    merged_queries = list(report.queries_used) + [q for q in extra.queries_used if q not in report.queries_used]

    # Cap total findings at _MAX_FINDINGS
    merged_findings = merged_findings[:_MAX_FINDINGS]

    logger.info("GAP fill: +%d findings from query %r", len(extra.findings), query[:60])
    return ResearchReport(
        findings=merged_findings,
        gaps=remaining_gaps,
        queries_used=merged_queries,
    )


# ── Orchestrator ─────────────────────────────────────────────


DEFAULT_RESEARCH_BUDGET = 7.0


def _combine_raw_snippets(search_results: list[tuple[str, str]]) -> str:
    """Concatenate raw snippets for the degraded-path injection."""
    return "\n\n".join(snippets for _, snippets in search_results)


async def run_research(
    ctx: ProxyContext,
    prompt: str,
    messages: list[dict] | None,
    *,
    total_budget: float = DEFAULT_RESEARCH_BUDGET,
    store_query_on_ctx: bool = False,
) -> ResearchReport | str:
    """End-to-end research pipeline.

    Returns a ``ResearchReport`` on the nominal path (with sourced findings),
    or a raw-snippets string when we fell back at any step. An empty string
    means the pipeline produced nothing usable (no search results at all).

    When ``store_query_on_ctx`` is True, the first planned query is stored on
    ``ctx.last_search_query`` so the FAKE tool_use fallback (proxy mode) can
    still surface a reasonable query if Serper is down.
    """
    budget = Budget(total_budget)

    # 1. PLAN
    queries = await plan_queries(ctx, prompt, messages, budget=budget)
    if not queries:
        return ""
    if store_query_on_ctx:
        ctx.last_search_query = " ".join(queries[:3])

    # 2. SEARCH
    search_results = await search_parallel(ctx, queries, budget=budget)
    if not search_results:
        logger.info("Research: no search results, returning empty")
        return ""

    # 3. READ + SYNTHESIZE
    report = await read_and_synthesize(ctx, prompt, search_results, budget=budget)

    # If READ degraded (zero findings + "synthesis failed" gap), return raw snippets
    if not report.findings:
        logger.info("Research: degraded to raw snippets (%.2fs left)", budget.remaining())
        return _combine_raw_snippets(search_results)

    # 4. GAP FILL (optional)
    report = await gap_fill(ctx, prompt, report, budget=budget)

    return report


def format_research_report(report: ResearchReport) -> str:
    """Render a ResearchReport as a readable, structured block for injection into the brain prompt.

    The format favours discrete sourced facts over paragraph prose so the brain
    can cite or ignore each finding independently.
    """
    lines = ["[Research findings — use as evidence, cite sources when relevant]"]
    for f in report.findings:
        lines.append(f"- FACT ({f.confidence}): {f.fact} [source: {f.source_url}]")
    if report.gaps:
        lines.append("Gaps: " + "; ".join(report.gaps))
    if report.queries_used:
        lines.append("Queries used: " + ", ".join(report.queries_used))
    return "\n".join(lines)
