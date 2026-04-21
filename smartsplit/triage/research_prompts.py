"""Prompts for the mini research agent (PLAN, SYNTHESIZE, GAP).

Kept in a dedicated module to keep ``research.py`` focused on orchestration.
All prompts force strict JSON output so they can be parsed with ``extract_json``.
"""

from __future__ import annotations

PLAN_PROMPT = """\
You are a research planner. Decompose the user's question into 2-3 short \
Google search queries that cover DIFFERENT angles (not rephrasings of each other).

Think about what types of sources would best answer the question:
- current versions / release notes → "<topic> 2025 release"
- benchmarks / comparisons → "<topic> benchmark vs <alternative>"
- concrete examples / github repos → "<topic> github example"
- official docs / specs → "<topic> official documentation"

Return ONLY a JSON array of query strings. No prose, no markdown fences.
Max 3 queries, each under 80 chars. Use the project context if provided.

Example output: ["llm routing frameworks 2025", "openai compatible proxy github", "llm gateway benchmark"]

{context}--- BEGIN USER PROMPT ---
{prompt}
--- END USER PROMPT ---"""


SYNTHESIZE_PROMPT = """\
You are a research analyst. Read the web search results below and extract \
VERIFIED facts that answer the user's question. Be strict: if a claim has no \
supporting URL in the results, DO NOT include it.

Return ONLY a JSON object with this exact shape:
{{
  "findings": [
    {{"fact": "short factual statement", "source_url": "https://...", "confidence": "high|medium|low"}}
  ],
  "gaps": ["specific information still missing, if any"]
}}

Rules:
- Each finding must cite an actual URL from the search results.
- confidence: "high" if multiple sources agree, "medium" if one clear source, "low" if only implied.
- facts must be short (one sentence max).
- gaps: list missing info that would improve the answer (empty list if satisfied).
- Max 6 findings. Skip duplicates and marketing fluff.

--- USER QUESTION ---
{prompt}

--- SEARCH RESULTS ---
{search_results}
--- END ---"""


GAP_PROMPT = """\
You are a research assistant. The previous search left these gaps:
{gaps}

Generate ONE short Google search query (under 80 chars) that would fill \
the most important gap. Return ONLY the query as a JSON string, no array, no prose.

Example output: "framework X streaming benchmark 2025"

--- ORIGINAL USER QUESTION ---
{prompt}
--- END ---"""
