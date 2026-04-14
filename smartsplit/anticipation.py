"""Tool anticipation — predict and pre-execute read-only tool calls (Agent mode).

In agent mode (request contains tools), SmartSplit predicts which read-only tools
the LLM will call next, pre-executes them, and injects the results into the
system message. This reduces round-trip latency in agentic loops.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING

from smartsplit.intention_detector import AnticipatedTool
from smartsplit.planner import _extract_json
from smartsplit.tool_pattern_learner import _extract_context_signals
from smartsplit.tool_registry import (
    FILE_REF_RE as _FILE_REF_RE,
)
from smartsplit.tool_registry import (
    GREP_TOOLS as _GREP_TOOLS,
)
from smartsplit.tool_registry import (
    LIST_DIR_TOOLS as _LIST_DIR_TOOLS,
)
from smartsplit.tool_registry import (
    READ_TOOLS as _READ_TOOLS,
)
from smartsplit.tool_registry import (
    SEARCH_TOOLS as _SEARCH_TOOLS,
)
from smartsplit.tool_registry import (
    WRITE_TOOLS as _WRITE_TOOLS,
)

if TYPE_CHECKING:
    from smartsplit.formats import OpenAIRequest
    from smartsplit.pipeline import ProxyContext

logger = logging.getLogger("smartsplit.anticipation")

_SEARCH_QUERY_PROMPT = """\
Extract 1-3 short Google search queries from this user prompt. \
Return ONLY a JSON array of query strings, nothing else.
Focus on what specific information the user needs from the web.
Use the project context below to make the queries specific and relevant.

Example: ["open source LLM routing frameworks 2025", "best AI proxy projects github"]

{context}--- BEGIN PROMPT ---
{prompt}
--- END PROMPT ---"""


def _extract_project_context(messages: list[dict]) -> str:
    """Extract project context from system messages (first 500 chars)."""
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 20:
                return "--- PROJECT CONTEXT ---\n" + content[:500] + "\n--- END CONTEXT ---\n\n"
    return ""


async def _fill_missing_args(
    ctx: ProxyContext,
    request_id: str,
    predictions: list,
    user_prompt: str,
    messages: list[dict],
) -> list:
    """Fill in missing args for anticipated tools.

    The free LLM predictor often predicts the tool type but omits args.
    We extract plausible args from the user prompt and context.
    """

    # Extract file paths mentioned in the user prompt
    mentioned_files = list(dict.fromkeys(_FILE_REF_RE.findall(user_prompt)))

    result = []
    for t in predictions:
        args = dict(t.args)

        if t.tool in _READ_TOOLS and not args.get("path"):
            if mentioned_files:
                # Create one prediction per mentioned file
                for path in mentioned_files[:3]:
                    result.append(
                        AnticipatedTool(
                            tool=t.tool, args={**args, "path": path}, reason=t.reason, confidence=t.confidence
                        )
                    )
                continue
            else:
                logger.debug("[%s] Skipping %s: no path in args and no files in prompt", request_id, t.tool)
                continue

        elif t.tool in _LIST_DIR_TOOLS and not args.get("path"):
            args["path"] = "."

        elif t.tool in _SEARCH_TOOLS and not args.get("query"):
            query = user_prompt[:200]
            try:
                context = _extract_project_context(messages)
                raw = await ctx.registry.call_free_llm(
                    _SEARCH_QUERY_PROMPT.replace("{context}", context).replace("{prompt}", user_prompt[:500]),
                    prefer="cerebras",
                )
                parsed_q = json.loads(_extract_json(raw))
                if isinstance(parsed_q, list) and parsed_q:
                    query = " ".join(str(q) for q in parsed_q[:3])
                    logger.info("[%s] Extracted search query: %r", request_id, query)
            except Exception:
                logger.debug("[%s] Search query extraction failed, using raw prompt", request_id)
            args["query"] = query

        elif t.tool in _GREP_TOOLS and not args.get("pattern"):
            logger.debug("[%s] Skipping %s: no pattern in args", request_id, t.tool)
            continue

        result.append(AnticipatedTool(tool=t.tool, args=args, reason=t.reason, confidence=t.confidence))

    return result


async def _run_anticipation(
    ctx: ProxyContext,
    request_id: str,
    messages_for_predict: list[dict],
    available_tools: list[dict] | None,
    user_prompt: str,
    all_messages: list[dict],
) -> list[dict[str, str]]:
    """Shared anticipation pipeline — predict, filter, execute, build context.

    Used by both anticipate_tools (OpenAI format) and anticipate_anthropic.
    Returns a list of {tool, args_summary, content} dicts for injection.
    """
    if not ctx.detector or not ctx.anticipator:
        return []

    prediction = await ctx.detector.predict(messages_for_predict, available_tools)

    if not prediction.should_anticipate or not prediction.tools:
        ctx.anticipation_stats["predictions_skipped"] += 1
        logger.info("[%s] No tools anticipated (confidence=%.2f)", request_id, prediction.confidence)
        return []

    ctx.anticipation_stats["predictions_made"] += 1
    ctx.anticipation_stats["tools_anticipated"] += len(prediction.tools)
    tool_names = [t.tool for t in prediction.tools]
    logger.info("[%s] Anticipated tools: %s (confidence=%.2f)", request_id, tool_names, prediction.confidence)

    # Record prediction for learning feedback
    if ctx.pattern_learner:
        ctx.pattern_learner.record_prediction(
            request_id=request_id,
            predicted_tools=[{"tool": t.tool, "args": dict(t.args)} for t in prediction.tools],
            context_signals=_extract_context_signals(messages_for_predict),
        )

    # Filter out files already in the conversation (already read or written)
    already_read = extract_already_read_paths(all_messages)
    recently_written = extract_recently_written_paths(all_messages)
    skip_paths = already_read | recently_written

    safe_predictions = [t for t in prediction.tools if t.args.get("path") not in skip_paths]
    if len(safe_predictions) < len(prediction.tools):
        skipped = len(prediction.tools) - len(safe_predictions)
        ctx.anticipation_stats["files_already_read"] += len(
            already_read & {t.args.get("path") for t in prediction.tools}
        )
        ctx.anticipation_stats["files_recently_written"] += len(
            recently_written & {t.args.get("path") for t in prediction.tools}
        )
        reasons = []
        if already_read:
            reasons.append("already read")
        if recently_written:
            reasons.append("recently written")
        logger.info("[%s] Skipped %d anticipated tools (%s)", request_id, skipped, ", ".join(reasons))

    if not safe_predictions:
        return []

    enriched_predictions = await _fill_missing_args(
        ctx, request_id, safe_predictions, user_prompt, messages_for_predict
    )

    to_execute = [AnticipatedTool(tool=t.tool, args=t.args) for t in enriched_predictions]
    results = await ctx.anticipator.execute(to_execute)

    context_parts: list[dict[str, str]] = []
    for r in results:
        if not r.success:
            continue
        args_summary = " ".join(str(k) + "=" + str(v) for k, v in r.args.items())
        context_parts.append({"tool": r.tool, "args_summary": args_summary, "content": r.content})

    return context_parts


async def anticipate_tools(
    ctx: ProxyContext,
    request_id: str,
    body_dict: dict,
    raw_messages: list[dict[str, str]],
    parsed: OpenAIRequest,
) -> list[dict[str, str]]:
    """Predict and pre-execute read-only tool calls (OpenAI format).

    Returns a list of {tool, args_summary, content} dicts for injection,
    or an empty list if nothing was anticipated.
    """
    messages_for_predict = body_dict.get("messages", [])
    user_prompt = ""
    for msg in reversed(messages_for_predict):
        if msg.get("role") == "user" and msg.get("content"):
            user_prompt = msg["content"]
            break

    return await _run_anticipation(
        ctx, request_id, messages_for_predict, parsed.tools, user_prompt, messages_for_predict
    )


def inject_anticipated_context(
    messages: list[dict],
    anticipated: list[dict[str, str]],
) -> list[dict]:
    """Inject anticipated tool results into the messages.

    Adds a system message block AFTER the existing system messages with
    the pre-collected context. Preserves all original messages intact.
    """
    if not anticipated:
        return messages

    parts = [
        "[SmartSplit — Pre-collected context]",
        "The following data was collected automatically. Use if relevant, ignore if not.",
        "",
    ]
    for item in anticipated:
        parts.append(f"--- {item['tool']}({item['args_summary']}) ---")
        parts.append(item["content"])
        parts.append("")
    parts.append("[End of SmartSplit context]")
    context_block = "\n".join(parts)

    # Inject as the FIRST system message (LLMs attend most to beginning and end,
    # "lost in the middle" effect means middle content gets ignored).
    enriched = copy.deepcopy(messages)
    enriched.insert(0, {"role": "system", "content": context_block})

    return enriched


def extract_already_read_paths(messages: list[dict]) -> set[str]:
    """Extract file paths already read in the conversation (no need to pre-fetch again)."""
    paths: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function", {})
            name = func.get("name", "") if isinstance(func, dict) else ""
            if name not in _READ_TOOLS:
                continue
            args_raw = func.get("arguments", "{}") if isinstance(func, dict) else "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except (json.JSONDecodeError, TypeError):
                args = {}
            path = args.get("path") or args.get("file_path") or args.get("filename", "")
            if path:
                paths.add(path)
    return paths


def extract_recently_written_paths(messages: list[dict]) -> set[str]:
    """Extract file paths that were written/edited recently in the conversation."""
    paths: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function", {})
            name = func.get("name", "") if isinstance(func, dict) else ""
            if name not in _WRITE_TOOLS:
                continue
            args_raw = func.get("arguments", "{}") if isinstance(func, dict) else "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except (json.JSONDecodeError, TypeError):
                args = {}
            path = args.get("path") or args.get("file_path") or args.get("filename", "")
            if path:
                paths.add(path)
    return paths


def extract_actual_tool_calls(messages: list[dict]) -> list[dict]:
    """Extract tool calls the LLM actually made from the conversation messages.

    Looks for assistant messages with tool_calls field. Returns a list of
    {tool, args} dicts for comparison with predictions.
    """
    calls: list[dict] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function", {})
            if not isinstance(func, dict):
                continue
            name = func.get("name", "")
            args_raw = func.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except (json.JSONDecodeError, TypeError):
                args = {}
            if name:
                calls.append({"tool": name, "args": args})
    return calls


async def anticipate_anthropic(
    ctx: ProxyContext,
    request_id: str,
    body_dict: dict,
) -> list[dict[str, str]]:
    """Run the anticipation pipeline for Anthropic-format requests.

    Converts to OpenAI format, then delegates to the shared pipeline.
    """
    from smartsplit.formats import anthropic_messages_to_openai, extract_anthropic_prompt

    if not ctx.detector or not ctx.anticipator:
        return []

    try:
        openai_body = anthropic_messages_to_openai(body_dict)
    except (AttributeError, TypeError, KeyError) as exc:
        logger.warning("[%s] Anthropic message conversion failed: %s: %s", request_id, type(exc).__name__, exc)
        return []

    messages_for_predict = openai_body.get("messages", [])
    user_prompt = extract_anthropic_prompt(body_dict)

    return await _run_anticipation(
        ctx, request_id, messages_for_predict, openai_body.get("tools"), user_prompt, messages_for_predict
    )
