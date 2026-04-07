#!/usr/bin/env python3
"""SmartSplit Provider Watch — daily check for LLM ecosystem changes.

Searches for:
1. New free LLM API providers
2. Benchmark changes (model rankings)
3. Free tier limit changes (rate limits, deprecations)
4. New models relevant to SmartSplit's task types

Outputs a Markdown report suitable for a GitHub issue.

Usage:
    # Requires SERPER_API_KEY and GROQ_API_KEY (or any LLM key)
    python scripts/provider_watch.py

    # Or via GitHub Action (see .github/workflows/provider-watch.yml)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone

import httpx


SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

CURRENT_PROVIDERS = [
    "Groq", "Cerebras", "Google Gemini", "DeepSeek", "OpenRouter",
    "Mistral", "Serper", "Tavily", "Anthropic Claude", "OpenAI GPT",
]

SEARCH_QUERIES = [
    "new free LLM API provider 2026 launched this week",
    "LLM benchmark update April 2026 code reasoning",
    "Groq Cerebras Gemini DeepSeek free tier changes 2026",
    "new open source LLM model release 2026 free API",
    "LLM API deprecation rate limit change 2026",
]

ANALYSIS_PROMPT = """\
You are a competitive intelligence analyst for SmartSplit, an LLM router that \
routes prompts to the cheapest capable LLM provider.

Our current providers: {providers}

Based on the search results below, write a concise Markdown report covering:

## New Providers
Any new free LLM API providers worth integrating? (name, free tier limits, strengths)

## Benchmark Changes
Have any models improved or degraded significantly? Should we update our competence scores?

## Free Tier Changes
Rate limit changes, deprecations, new restrictions on our current providers?

## New Models
New model releases that outperform what we currently use for specific tasks (code, reasoning, translation, etc.)?

## Action Items
Bullet list of concrete changes to make in SmartSplit (update scores, add provider, etc.)
If nothing changed, say "No action needed."

Search results:
{results}

Be concise. Only report things that are NEW or CHANGED. Skip anything we already know.\
"""


async def search(http: httpx.AsyncClient, query: str) -> list[dict]:
    """Run a web search via Serper."""
    response = await http.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_API_KEY},
        json={"q": query, "num": 5},
    )
    response.raise_for_status()
    return [
        {"title": r.get("title", ""), "snippet": r.get("snippet", ""), "link": r.get("link", "")}
        for r in response.json().get("organic", [])[:5]
    ]


async def analyze(http: httpx.AsyncClient, search_results: str) -> str:
    """Use Groq to analyze the search results."""
    prompt = ANALYSIS_PROMPT.format(
        providers=", ".join(CURRENT_PROVIDERS),
        results=search_results,
    )
    response = await http.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2000,
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


async def main() -> str:
    if not SERPER_API_KEY:
        print("ERROR: SERPER_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    async with httpx.AsyncClient(timeout=30.0) as http:
        # Run all searches concurrently
        all_results = await asyncio.gather(*(search(http, q) for q in SEARCH_QUERIES))

        # Format results
        formatted = ""
        for query, results in zip(SEARCH_QUERIES, all_results):
            formatted += f"\n### Query: {query}\n"
            for r in results:
                formatted += f"- **{r['title']}**: {r['snippet']} ({r['link']})\n"

        # Analyze with LLM
        report = await analyze(http, formatted)

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    header = f"# Provider Watch Report — {date}\n\n"
    return header + report


if __name__ == "__main__":
    report = asyncio.run(main())
    print(report)

    # Write to file for GitHub Action
    output_path = os.environ.get("REPORT_OUTPUT", "")
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
