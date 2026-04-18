"""Enrichment pipeline — workers do prep work, then brain synthesizes.

Handles the ENRICH path: web search, pre-analysis, multi-perspective comparison,
and context summary. Results are injected into the brain's prompt as additional context.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from typing import TYPE_CHECKING

from smartsplit.json_utils import extract_json
from smartsplit.models import (
    Complexity,
    RouteResult,
    Subtask,
    TaskType,
    TerminationState,
)
from smartsplit.tools.anticipation import SEARCH_QUERY_PROMPT, extract_project_context

if TYPE_CHECKING:
    from smartsplit.proxy.pipeline import ProxyContext

logger = logging.getLogger("smartsplit.enrichment")


async def _extract_search_query(
    ctx: ProxyContext,
    prompt: str,
    messages: list[dict[str, str]] | None,
    *,
    store_on_ctx: bool = False,
) -> str:
    """Refine ``prompt`` into a concise Google-ready query via free LLM, else return it unchanged."""
    try:
        context = extract_project_context(messages or [])
        raw_queries = await ctx.registry.call_free_llm(
            SEARCH_QUERY_PROMPT.replace("{context}", context).replace("{prompt}", prompt),
            prefer="cerebras",
        )
        parsed = json.loads(extract_json(raw_queries))
        if isinstance(parsed, list) and parsed:
            search_prompt = " ".join(str(q) for q in parsed[:3])
            if store_on_ctx:
                # Expose the refined query for FAKE tool_use fallback when Serper fails.
                ctx.last_search_query = search_prompt
            logger.info("Search query extracted: %r", search_prompt)
            return search_prompt
    except Exception as e:
        logger.debug("Search query extraction failed: %s, using raw prompt", type(e).__name__)
    return prompt


_ENRICHMENT_PROMPTS: dict[str, str] = {
    "web_search": "{prompt}",
    "pre_analysis": (
        "Analyze this request and provide structured context that would help "
        "another AI give a better response. Identify key concepts, constraints, "
        "and relevant background information.\n\nRequest: {prompt}"
    ),
    "multi_perspective": (
        "List the main options/alternatives and their key pros and cons "
        "for this decision. Be factual and concise.\n\nQuestion: {prompt}"
    ),
    "context_summary": (
        "Summarize this conversation history into key points, decisions made, "
        "and current state. Be concise.\n\nConversation:\n{context}"
    ),
}


def _build_enrichment_subtasks(
    prompt: str,
    enrichment_types: list[str],
    messages: list[dict[str, str]] | None = None,
) -> list[Subtask]:
    """Build worker subtasks for each enrichment type."""
    subtasks: list[Subtask] = []
    for etype in enrichment_types:
        template = _ENRICHMENT_PROMPTS.get(etype)
        if not template:
            continue

        if etype == "web_search":
            subtasks.append(
                Subtask(
                    type=TaskType.WEB_SEARCH,
                    content=prompt,
                    complexity=Complexity.LOW,
                )
            )
        elif etype == "context_summary":
            context = "\n".join(f"[{m['role']}]: {m.get('content', '')[:200]}" for m in (messages or []))
            subtasks.append(
                Subtask(
                    type=TaskType.SUMMARIZE,
                    content=template.replace("{context}", context),
                    complexity=Complexity.LOW,
                )
            )
        else:
            subtasks.append(
                Subtask(
                    type=TaskType.REASONING,
                    content=template.replace("{prompt}", prompt),
                    complexity=Complexity.MEDIUM,
                )
            )
    return subtasks


def build_enriched_messages(
    original_messages: list[dict[str, str]],
    prompt: str,
    worker_results: list[RouteResult],
) -> list[dict[str, str]]:
    """Build the enriched message list: original messages + worker context injected.

    The original conversation is kept intact. Worker results are injected as
    additional context in the last user message. The brain sees:
    - All system prompts unchanged
    - All conversation history unchanged
    - Last user message = original prompt + worker context block
    """
    if not worker_results:
        return original_messages

    # Build the context block from worker results
    context_parts = []
    for r in worker_results:
        if r.response and r.termination == TerminationState.COMPLETED:
            label = r.type.value.replace("_", " ").title()
            context_parts.append(f"- {label}: {r.response}")

    if not context_parts:
        return original_messages

    context_block = (
        "\n\n[Additional context gathered by SmartSplit — use if relevant, ignore if not. "
        "IMPORTANT: Always respond in the same language as the user's message above.]\n" + "\n".join(context_parts)
    )

    # Inject into the last user message (deep copy to protect originals for fallback).
    # Preserve all original keys (tool_calls, tool_call_id, name) and append context to
    # content — string content gets concatenated, list content (multimodal) gets a new
    # text part appended.
    enriched = [copy.deepcopy(m) for m in original_messages]
    injected = False
    for i in range(len(enriched) - 1, -1, -1):
        if enriched[i]["role"] == "user":
            content = enriched[i].get("content")
            if isinstance(content, list):
                enriched[i]["content"] = content + [{"type": "text", "text": context_block}]
            else:
                enriched[i]["content"] = (content or "") + context_block
            injected = True
            break

    if not injected:
        logger.warning("No user message found; appending enrichment as new user message")
        enriched.append({"role": "user", "content": context_block})

    return enriched


async def enrich_and_forward(
    ctx: ProxyContext,
    prompt: str,
    enrichment_types: list[str],
    messages: list[dict[str, str]] | None = None,
) -> tuple[str | None, list[RouteResult]]:
    """ENRICH path — workers do prep work, then brain synthesizes."""
    logger.info("ENRICH (%s) → workers, then brain: %s", enrichment_types, ctx.registry.brain_name)

    search_prompt = await _extract_search_query(ctx, prompt, messages) if "web_search" in enrichment_types else prompt

    # Build and execute worker subtasks
    worker_subtasks = _build_enrichment_subtasks(search_prompt, enrichment_types, messages)

    worker_results: list[RouteResult] = []
    if worker_subtasks:
        raw = await asyncio.gather(*(ctx.router.route(st, ctx.mode) for st in worker_subtasks))
        worker_results = [r for r in raw if r.response]
        logger.info("Workers completed: %s/%s succeeded", len(worker_results), len(worker_subtasks))
        if logger.isEnabledFor(logging.DEBUG):
            for r in worker_results:
                logger.debug("Worker [%s] via %s: %s", r.type.value, r.provider, r.response[:200] if r.response else "")

    # Build enriched messages and forward to brain
    enriched_messages = build_enriched_messages(
        messages or [{"role": "user", "content": prompt}],
        prompt,
        worker_results,
    )

    try:
        content, usage = await ctx.registry.call_brain(prompt, messages=enriched_messages)
        brain_result = RouteResult(
            type=TaskType.GENERAL,
            response=content,
            provider=ctx.registry.brain_name,
            termination=TerminationState.COMPLETED,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        all_results = worker_results + [brain_result]
        return content, all_results
    except Exception as e:
        logger.warning("Brain failed with enriched prompt: %s, retrying without enrichment", type(e).__name__)
        # Retry brain with original (non-enriched) messages
        try:
            original_messages = messages or [{"role": "user", "content": prompt}]
            content, usage = await ctx.registry.call_brain(prompt, messages=original_messages)
            brain_result = RouteResult(
                type=TaskType.GENERAL,
                response=content,
                provider=ctx.registry.brain_name,
                termination=TerminationState.ESCALATED,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
            return content, worker_results + [brain_result]
        except Exception:
            logger.error("Brain failed on both enriched and non-enriched attempts")
            return None, []


async def enrich_only(
    ctx: ProxyContext,
    prompt: str,
    enrichment_types: list[str],
    messages: list[dict[str, str]] | None = None,
) -> list[RouteResult]:
    """Run enrichment workers only — no brain call.

    Used in agent mode where the brain is the client's own LLM (passthrough).
    Returns worker results for injection into the request context.
    Stores extracted search query on ctx.last_search_query for FAKE tool_use fallback.
    """
    logger.info("ENRICH workers only (%s)", enrichment_types)

    search_prompt = (
        await _extract_search_query(ctx, prompt, messages, store_on_ctx=True)
        if "web_search" in enrichment_types
        else prompt
    )

    worker_subtasks = _build_enrichment_subtasks(search_prompt, enrichment_types, messages)

    if not worker_subtasks:
        return []

    raw = await asyncio.gather(*(ctx.router.route(st, ctx.mode) for st in worker_subtasks))
    worker_results = [r for r in raw if r.response]
    logger.info("Workers completed: %s/%s succeeded", len(worker_results), len(worker_subtasks))
    for r in worker_results:
        logger.debug("Worker [%s] via %s: %s", r.type.value, r.provider, r.response[:200] if r.response else "")
    return worker_results
