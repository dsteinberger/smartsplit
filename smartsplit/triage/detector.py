"""Request triage — decides TRANSPARENT or ENRICH for each incoming request.

Two-phase detection:
  1. Fast keyword heuristic (<1ms) — catches obvious cases
  2. LLM classification (~400ms) — catches intent that keywords miss

Every request is classified into one of two modes:
  TRANSPARENT — forward directly to the brain (main LLM), zero overhead
  ENRICH     — workers do prep work (search, analysis), then brain synthesizes
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import TYPE_CHECKING

from smartsplit.json_utils import extract_json
from smartsplit.triage.i18n_keywords import ANALYSIS_KEYWORDS_I18N, COMPARISON_KEYWORDS_I18N
from smartsplit.triage.planner import detect_domains

if TYPE_CHECKING:
    from smartsplit.providers.registry import ProviderRegistry

logger = logging.getLogger("smartsplit.detector")

# Minimum prompt length worth enriching
_ENRICH_MIN_CHARS = 80

# Conversation size thresholds for context summary enrichment
_LONG_HISTORY_MESSAGES = 10
_LONG_HISTORY_CHARS = 5000

# Keywords that signal enrichment opportunities
_COMPARISON_KEYWORDS_BASE = [
    " vs ",
    " versus ",
    "compare",
    "which is better",
    "pros and cons",
    "advantages",
    "disadvantages",
    "tradeoff",
    "trade-off",
]
_ANALYSIS_KEYWORDS_BASE = [
    "refactor",
    "review",
    "analyze",
    "audit",
    "explain this",
    "what's wrong",
    "improve",
    "optimize",
    "diagnose",
]


def _merge_i18n_keywords(base: list[str], i18n: dict[str, list[str]]) -> list[str]:
    """Flatten multilingual keyword lists onto the base, deduplicated while preserving order."""
    merged = list(base)
    for lang_kws in i18n.values():
        merged.extend(lang_kws)
    return list(dict.fromkeys(merged))


_COMPARISON_KEYWORDS = _merge_i18n_keywords(_COMPARISON_KEYWORDS_BASE, COMPARISON_KEYWORDS_I18N)
_ANALYSIS_KEYWORDS = _merge_i18n_keywords(_ANALYSIS_KEYWORDS_BASE, ANALYSIS_KEYWORDS_I18N)


class TriageDecision(StrEnum):
    """Request triage outcome: TRANSPARENT (direct to brain) or ENRICH (workers prep first)."""

    TRANSPARENT = "transparent"  # Forward direct to brain
    ENRICH = "enrich"  # Workers prep + brain synthesizes


def _has_tool_messages(messages: list[dict[str, str]]) -> bool:
    """Check if the conversation contains tool calls (agentic flow)."""
    return any(m.get("role") in ("tool", "function") for m in messages)


def detect(
    prompt: str,
    messages: list[dict[str, str]] | None = None,
) -> tuple[TriageDecision, list[str]]:
    """Fast heuristic detector — decides TRANSPARENT or ENRICH without LLM calls.

    Returns (decision, enrichment_types) where enrichment_types is a list of
    enrichment categories to run (e.g. ["web_search", "pre_analysis"]).
    """
    if not prompt:
        return TriageDecision.TRANSPARENT, []

    msgs = messages or []
    enrichments: list[str] = []

    # Tool calls in conversation → agentic flow → never interfere
    if _has_tool_messages(msgs):
        return TriageDecision.TRANSPARENT, []

    # ── Early checks (before length filter) ──────────────────

    # Long conversation history → context summary to save brain tokens
    # Only count user/assistant messages — system messages often contain large
    # injected files (IDE context) which don't indicate a long conversation.
    # Strip XML-tagged metadata from char count (agents inject large blocks).
    if msgs:
        from smartsplit.proxy.formats import strip_agent_metadata

        conversation_msgs = [m for m in msgs if m.get("role") in ("user", "assistant")]
        total_chars = sum(len(strip_agent_metadata(m.get("content", ""))) for m in conversation_msgs)
        if len(conversation_msgs) > _LONG_HISTORY_MESSAGES or total_chars > _LONG_HISTORY_CHARS:
            enrichments.append("context_summary")

    # Web search / current data detection — checked early because even
    # short prompts like "latest F1 results 2026?" deserve web enrichment
    domains = detect_domains(prompt)
    domain_names = [d for d, _ in domains]
    if "web_search" in domain_names:
        enrichments.append("web_search")

    # Short prompts → not worth enriching (unless already flagged above)
    if len(prompt.strip()) < _ENRICH_MIN_CHARS and not enrichments:
        return TriageDecision.TRANSPARENT, []

    prompt_lower = prompt.lower()

    # Comparison / decision prompts
    if any(kw in prompt_lower for kw in _COMPARISON_KEYWORDS):
        enrichments.append("multi_perspective")

    # Complex analysis prompts
    if any(kw in prompt_lower for kw in _ANALYSIS_KEYWORDS) and len(prompt.strip()) > 200:
        enrichments.append("pre_analysis")

    # Multi-domain prompt (code + translation, math + writing, etc.)
    if len(domain_names) >= 2 and len(prompt.strip()) > 200 and "pre_analysis" not in enrichments:
        enrichments.append("pre_analysis")

    if enrichments:
        logger.debug("Triage → ENRICH (reasons: %s, domains: %s)", enrichments, domain_names)
        return TriageDecision.ENRICH, enrichments

    logger.debug("Triage → TRANSPARENT (domains: %s, prompt_len: %d)", domain_names, len(prompt.strip()))
    return TriageDecision.TRANSPARENT, []


# Minimum prompt length to justify an LLM classification call
LLM_DETECT_MIN_CHARS = 40

_TRIAGE_PROMPT = """\
You are a request triage engine. Decide if the user's prompt needs external enrichment.

Return ONLY a JSON object with:
- "decision": "transparent" (forward directly) or "enrich" (needs worker prep)
- "enrichments": list of enrichment types needed (empty if transparent)

Valid enrichment types:
- "web_search" — the answer would be BETTER with real external data (projects, tools, docs, news, versions, benchmarks, examples)
- "multi_perspective" — prompt asks to COMPARE options or weigh pros/cons
- "pre_analysis" — prompt requires deep analysis of complex code/architecture

Rules:
- MOST prompts are "transparent" — only enrich when it clearly adds value
- "web_search" when real-world data improves the answer: finding projects, tools, libraries, current versions, news, benchmarks, documentation
- NOT web_search: explaining concepts, writing code, answering from general knowledge, simple opinions
- Code tasks, explanations, and simple questions are always transparent

Respond with ONLY the JSON object, nothing else.
Example: {"decision": "enrich", "enrichments": ["web_search"]}
Example: {"decision": "transparent", "enrichments": []}

--- BEGIN PROMPT ---
"""


async def detect_with_llm(
    prompt: str,
    registry: ProviderRegistry,
) -> tuple[TriageDecision, list[str]]:
    """LLM-based triage for prompts that keywords missed.

    Uses a fast free LLM to decide if enrichment adds value.
    Returns TRANSPARENT if the LLM call fails or finds nothing enrichment-worthy.
    """
    try:
        raw = await registry.call_free_llm(
            _TRIAGE_PROMPT + prompt + "\n--- END PROMPT ---",
            prefer="cerebras",
        )
        parsed = json.loads(extract_json(raw))
        if not isinstance(parsed, dict):
            return TriageDecision.TRANSPARENT, []

        decision = parsed.get("decision", "transparent")
        enrichments = parsed.get("enrichments", [])
        valid_enrichments = [e for e in enrichments if e in ("web_search", "multi_perspective", "pre_analysis")]

        if decision == "enrich" and valid_enrichments:
            logger.info("LLM triage → enrich (%s)", valid_enrichments)
            return TriageDecision.ENRICH, valid_enrichments

    except Exception as e:
        logger.debug("LLM triage failed: %s", type(e).__name__)

    return TriageDecision.TRANSPARENT, []
