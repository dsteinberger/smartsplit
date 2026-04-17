#!/usr/bin/env python3
"""SmartSplit Provider Watch — daily check for LLM ecosystem changes.

Searches for:
1. New free LLM API providers
2. Benchmark changes (model rankings)
3. Free tier limit changes (rate limits, deprecations)
4. New models relevant to SmartSplit's task types

Outputs a Markdown report suitable for a GitHub issue.

Usage:
    # Requires SERPER_API_KEY and (CEREBRAS_API_KEY or GROQ_API_KEY)
    python scripts/provider_watch.py

    # Or via GitHub Action (see .github/workflows/provider-watch.yml)
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

# ── Env vars ────────────────────────────────────────────────

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")

# ── Config reader ───────────────────────────────────────────

# Add repo root to sys.path once at module level
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_config_cache: dict | None = None


def _get_providers_config() -> dict:
    """Import and cache DEFAULT_PROVIDERS from config.py."""
    global _config_cache  # noqa: PLW0603
    if _config_cache is None:
        try:
            from smartsplit.config import DEFAULT_PROVIDERS

            _config_cache = dict(DEFAULT_PROVIDERS)
        except Exception:
            _config_cache = {}
    return _config_cache


def _read_current_config() -> str:
    """Format current config as text for the LLM prompt."""
    providers = _get_providers_config()
    if not providers:
        return "Could not read config"
    lines = []
    for name, cfg in providers.items():
        model = cfg.get("model", "")
        ptype = cfg.get("type", "free")
        fast = cfg.get("fast_model", "")
        strong = cfg.get("strong_model", "")
        limits = cfg.get("limits", {})
        desc = f"{name} (type={ptype}, model={model}"
        if fast:
            desc += f", fast={fast}"
        if strong:
            desc += f", strong={strong}"
        if limits:
            desc += f", limits={limits}"
        desc += ")"
        lines.append(desc)
    return "\n".join(lines)


# ── Health check ────────────────────────────────────────────

# Each entry: (name, url, model, env_var, auth_style)
# auth_style: "bearer" | "x-api-key" | "query"
HEALTH_CHECK_PROVIDERS = [
    ("groq", "https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile", "GROQ_API_KEY", "bearer"),
    (
        "cerebras",
        "https://api.cerebras.ai/v1/chat/completions",
        "qwen-3-235b-a22b-instruct-2507",
        "CEREBRAS_API_KEY",
        "bearer",
    ),
    (
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
        "query",
    ),
    (
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        "qwen/qwen3-coder:free",
        "OPENROUTER_API_KEY",
        "bearer",
    ),
    ("mistral", "https://api.mistral.ai/v1/chat/completions", "mistral-small-latest", "MISTRAL_API_KEY", "bearer"),
    (
        "huggingface",
        "https://router.huggingface.co/v1/chat/completions",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "HF_TOKEN",
        "bearer",
    ),
    ("serper", "https://google.serper.dev/search", None, "SERPER_API_KEY", "x-api-key"),
]

_HEALTH_CHECK_TIMEOUT = 15.0  # per-provider timeout


@dataclass
class HealthResult:
    name: str
    status: str  # "ok" | "error" | "skipped"
    reason: str = ""
    code: int = 0
    rate_limits: dict[str, str] = field(default_factory=dict)


async def health_check_one(
    http: httpx.AsyncClient,
    name: str,
    url: str,
    model: str | None,
    env_var: str,
    auth_style: str,
) -> HealthResult:
    """Ping one provider with a minimal request."""
    api_key = os.environ.get(env_var, "")
    if not api_key:
        return HealthResult(name=name, status="skipped", reason=f"{env_var} not set")

    try:
        if auth_style == "x-api-key":
            response = await http.post(
                url,
                headers={"X-API-KEY": api_key},
                json={"q": "test", "num": 1},
                timeout=_HEALTH_CHECK_TIMEOUT,
            )
        elif auth_style == "query":
            full_url = url.format(model=model, key=api_key)
            response = await http.post(
                full_url,
                json={"contents": [{"parts": [{"text": "Say OK"}]}], "generationConfig": {"maxOutputTokens": 5}},
                timeout=_HEALTH_CHECK_TIMEOUT,
            )
        else:
            response = await http.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5},
                timeout=_HEALTH_CHECK_TIMEOUT,
            )

        rate_limits = {h: response.headers[h] for h in response.headers if "ratelimit" in h.lower()}

        if response.status_code == 200:
            return HealthResult(name=name, status="ok", rate_limits=rate_limits)
        return HealthResult(
            name=name,
            status="error",
            code=response.status_code,
            reason=response.text[:200],
            rate_limits=rate_limits,
        )

    except Exception as e:
        return HealthResult(name=name, status="error", reason=str(e)[:200])


async def health_check_all(http: httpx.AsyncClient) -> str:
    """Run health checks on all providers concurrently."""
    tasks = [
        health_check_one(http, name, url, model, env_var, auth_style)
        for name, url, model, env_var, auth_style in HEALTH_CHECK_PROVIDERS
    ]
    results = await asyncio.gather(*tasks)

    lines = ["## Provider Health Check\n"]
    lines.append("| Provider | Status | Rate Limits |")
    lines.append("|----------|--------|-------------|")

    has_issues = False
    for r in results:
        rl_str = ", ".join(f"`{k}: {v}`" for k, v in r.rate_limits.items()) if r.rate_limits else "—"
        if r.status == "ok":
            lines.append(f"| {r.name} | :white_check_mark: OK | {rl_str} |")
        elif r.status == "skipped":
            lines.append(f"| {r.name} | :heavy_minus_sign: Skipped ({r.reason}) | — |")
        else:
            has_issues = True
            detail = r.reason or f"HTTP {r.code}"
            lines.append(f"| {r.name} | :x: FAIL — {detail} | {rl_str} |")

    lines.append("")
    if has_issues:
        lines.append(":warning: **Some providers failed health check — see details above.**\n")
    else:
        lines.append("All configured providers are healthy.\n")

    return "\n".join(lines)


# ── Web search ──────────────────────────────────────────────

SEARCH_QUERIES = [
    "new free LLM API provider 2026 launched this week",
    "LLM benchmark update April 2026 code reasoning",
    "Groq Cerebras Gemini DeepSeek HuggingFace Cloudflare free tier changes 2026",
    "new open source LLM model release 2026 free API",
    "LLM API deprecation rate limit change 2026",
    "Claude Opus Sonnet Haiku new model release 2026",
    "OpenAI GPT new model release 2026 o3 gpt-4o",
]


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


# ── LLM analysis ───────────────────────────────────────────

ANALYSIS_PROMPT = """\
You are a competitive intelligence analyst for SmartSplit, an LLM router that \
routes prompts to the best free cloud LLM API for each task (code, reasoning, translation, etc.).

SmartSplit is a CLOUD API ROUTER — it calls remote HTTP APIs. It does NOT run models locally.

Our current providers and EXACT configuration (read from code):
{providers}

CRITICAL RULES — do NOT recommend:
- Local/self-hosted solutions (Ollama, vLLM, llama.cpp, etc.) — we only use cloud APIs
- Providers we already have (check the list above — e.g. "Google AI Studio" = Gemini, which we have)
- Providers without a free tier or public API
- Models that require custom infrastructure to run
- Models or versions we ALREADY use — check the exact model IDs above before recommending anything
- Changes we already made — if our config already has the latest model, do NOT suggest updating to it

Based on the search results below, write a concise Markdown report covering:

## New Providers
Any NEW free cloud LLM API providers worth integrating? Must have: a free tier, an HTTP API, \
and be relevant to code/reasoning/translation/search tasks. Include free tier limits and strengths.

## Benchmark Changes
Have any models from our CURRENT providers improved or degraded significantly? \
Should we update our competence scores? Include specific benchmark numbers.

## Free Tier Changes
Rate limit changes, deprecations, new restrictions on our CURRENT providers?

## New Models
New model releases available via our CURRENT providers or new cloud APIs \
that outperform what we currently use?

## Model Tier Updates
Check our EXACT current models above before answering — only suggest changes \
if the model ID would actually change.

## Action Items
Bullet list of concrete changes to make in SmartSplit (update scores, add provider, etc.)
If nothing actionable, say "No action needed."

Search results:
{results}

Be concise and precise. Only report things that are NEW, CHANGED, and ACTIONABLE.\
"""


async def _call_llm(http: httpx.AsyncClient, url: str, key: str, model: str, prompt: str) -> str:
    """Call an OpenAI-compatible LLM endpoint."""
    response = await http.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2000,
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


async def analyze(http: httpx.AsyncClient, search_results: str) -> str:
    """Analyze search results. Tries Cerebras (235B) first, falls back to Groq (70B)."""
    prompt = ANALYSIS_PROMPT.format(
        providers=_read_current_config(),
        results=search_results,
    )

    # Cerebras (Qwen 235B) — better instruction following
    if CEREBRAS_API_KEY:
        try:
            return await _call_llm(
                http,
                "https://api.cerebras.ai/v1/chat/completions",
                CEREBRAS_API_KEY,
                "qwen-3-235b-a22b-instruct-2507",
                prompt,
            )
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError) as e:
            print(f"WARNING: Cerebras failed, falling back to Groq: {e}", file=sys.stderr)

    # Groq (LLaMA 70B) — fallback
    if not GROQ_API_KEY:
        return "*Analysis unavailable — both Cerebras and Groq failed or unconfigured.*"
    return await _call_llm(
        http,
        "https://api.groq.com/openai/v1/chat/completions",
        GROQ_API_KEY,
        "llama-3.3-70b-versatile",
        prompt,
    )


# ── Validation ──────────────────────────────────────────────


def _validate_report(analysis: str) -> str:
    """Append auto-validation notes for false positives."""
    providers = _get_providers_config()
    if not providers:
        return analysis

    known_providers = set(providers.keys())
    known_aliases = {
        "google ai studio": "gemini",
        "google gemini": "gemini",
        "claude": "anthropic",
        "gpt": "openai",
        "gpt-4o": "openai",
        "workers ai": "cloudflare",
    }
    known_models: set[str] = set()
    for cfg in providers.values():
        for key in ("model", "fast_model", "strong_model"):
            val = cfg.get(key, "")
            if val:
                known_models.add(val.lower())

    lower = analysis.lower()
    warnings: list[str] = []

    for provider in known_providers:
        if f"adding {provider}" in lower or f"add {provider}" in lower:
            warnings.append(f"- ~~Add {provider}~~ — already configured")
    for alias, real in known_aliases.items():
        if alias in lower and "add" in lower:
            warnings.append(f"- ~~Add {alias}~~ — already configured as `{real}`")

    for model in known_models:
        # Match on the first 3 segments of the model ID (e.g. "claude-sonnet-4")
        segments = model.split("/")[-1].split("-")[0:3]
        short = "-".join(segments)
        if short and short in lower and "update" in lower:
            warnings.append(f"- Note: `{model}` is already in our config")

    if warnings:
        return analysis + "\n\n## Auto-Validation Notes\n" + "\n".join(warnings)
    return analysis


# ── Main ────────────────────────────────────────────────────


async def main() -> str:
    if not SERPER_API_KEY:
        print("ERROR: SERPER_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    if not GROQ_API_KEY and not CEREBRAS_API_KEY:
        print("ERROR: GROQ_API_KEY or CEREBRAS_API_KEY required", file=sys.stderr)
        sys.exit(1)

    async with httpx.AsyncClient(timeout=30.0) as http:
        # Run health checks and searches concurrently
        health_task = health_check_all(http)
        search_tasks = [search(http, q) for q in SEARCH_QUERIES]
        health_report, *all_results = await asyncio.gather(health_task, *search_tasks)

        # Format search results
        formatted = ""
        for query, results in zip(SEARCH_QUERIES, all_results, strict=False):
            formatted += f"\n### Query: {query}\n"
            for r in results:
                formatted += f"- **{r['title']}**: {r['snippet']} ({r['link']})\n"

        # Analyze with LLM + validate
        analysis = _validate_report(await analyze(http, formatted))

    date = datetime.now(UTC).strftime("%Y-%m-%d")
    header = f"# Provider Watch Report — {date}\n\n"
    return header + health_report + "\n---\n\n" + analysis


if __name__ == "__main__":
    report = asyncio.run(main())
    print(report)

    output_path = os.environ.get("REPORT_OUTPUT", "")
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
