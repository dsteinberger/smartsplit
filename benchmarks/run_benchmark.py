"""SmartSplit Benchmark — Compare decompose+route vs single-model baselines.

Usage:
    # Start SmartSplit first:
    smartsplit

    # Then run the benchmark:
    python benchmarks/run_benchmark.py

    # Or with options:
    python benchmarks/run_benchmark.py --judge --baselines groq,gemini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path

import httpx

# ── Config ────────────────────────────────────────────────────

DEFAULT_PROXY_URL = "http://127.0.0.1:8420"
DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Baseline providers: name -> (url, headers_factory, body_factory)
# These call the providers directly (not through SmartSplit).


def _groq_call(prompt: str, api_key: str) -> dict:
    return {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "headers": {"Authorization": f"Bearer {api_key}"},
        "body": {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 4096,
        },
    }


def _gemini_call(prompt: str, api_key: str) -> dict:
    return {
        "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
        "headers": {"Content-Type": "application/json"},
        "body": {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096},
        },
    }


def _cerebras_call(prompt: str, api_key: str) -> dict:
    return {
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "headers": {"Authorization": f"Bearer {api_key}"},
        "body": {
            "model": "qwen-3-235b-a22b-instruct-2507",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 4096,
        },
    }


def _openrouter_call(prompt: str, api_key: str) -> dict:
    return {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {"Authorization": f"Bearer {api_key}"},
        "body": {
            "model": "qwen/qwen3-coder:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 4096,
        },
    }


def _mistral_call(prompt: str, api_key: str) -> dict:
    return {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "headers": {"Authorization": f"Bearer {api_key}"},
        "body": {
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 4096,
        },
    }


BASELINE_FACTORIES = {
    "groq": ("GROQ_API_KEY", _groq_call),
    "gemini": ("GEMINI_API_KEY", _gemini_call),
    "cerebras": ("CEREBRAS_API_KEY", _cerebras_call),
    "openrouter": ("OPENROUTER_API_KEY", _openrouter_call),
    "mistral": ("MISTRAL_API_KEY", _mistral_call),
}


# ── Data structures ───────────────────────────────────────────


@dataclass
class BenchmarkResult:
    prompt_id: str
    category: str
    method: str  # "smartsplit" or provider name
    response: str
    latency_ms: int
    estimated_tokens: int
    success: bool
    error: str = ""
    judge_score: float = 0.0


@dataclass
class BenchmarkReport:
    timestamp: str
    total_prompts: int
    methods: list[str]
    results: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# ── Core benchmark logic ──────────────────────────────────────


async def call_smartsplit(
    http: httpx.AsyncClient,
    prompt: str,
    proxy_url: str = DEFAULT_PROXY_URL,
    messages: list[dict[str, str]] | None = None,
) -> tuple[str, int]:
    """Send a prompt through the SmartSplit proxy (OpenAI format)."""
    body = {
        "model": "smartsplit",
        "messages": messages or [{"role": "user", "content": prompt}],
    }
    response = await http.post(
        f"{proxy_url}/v1/chat/completions",
        json=body,
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", len(content) // 4)
    return content, tokens


async def call_baseline_groq(
    http: httpx.AsyncClient,
    prompt: str,
    api_key: str,
) -> tuple[str, int]:
    """Call Groq directly."""
    cfg = _groq_call(prompt, api_key)
    response = await http.post(
        cfg["url"],
        headers=cfg["headers"],
        json=cfg["body"],
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", len(content) // 4)
    return content, tokens


async def call_baseline_gemini(
    http: httpx.AsyncClient,
    prompt: str,
    api_key: str,
) -> tuple[str, int]:
    """Call Gemini directly."""
    cfg = _gemini_call(prompt, api_key)
    response = await http.post(
        cfg["url"],
        headers=cfg["headers"],
        json=cfg["body"],
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    content = data["candidates"][0]["content"]["parts"][0]["text"]
    tokens = data.get("usageMetadata", {}).get("totalTokenCount", len(content) // 4)
    return content, tokens


async def call_baseline_cerebras(
    http: httpx.AsyncClient,
    prompt: str,
    api_key: str,
) -> tuple[str, int]:
    """Call Cerebras directly."""
    cfg = _cerebras_call(prompt, api_key)
    response = await http.post(
        cfg["url"],
        headers=cfg["headers"],
        json=cfg["body"],
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", len(content) // 4)
    return content, tokens


async def call_baseline_openai_compat(
    http: httpx.AsyncClient,
    prompt: str,
    api_key: str,
    factory_fn: callable,
) -> tuple[str, int]:
    """Generic caller for any OpenAI-compatible provider."""
    cfg = factory_fn(prompt, api_key)
    response = await http.post(
        cfg["url"],
        headers=cfg["headers"],
        json=cfg["body"],
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", len(content) // 4)
    return content, tokens


BASELINE_CALLERS = {
    "groq": call_baseline_groq,
    "gemini": call_baseline_gemini,
    "cerebras": call_baseline_cerebras,
    "openrouter": lambda http, prompt, key: call_baseline_openai_compat(http, prompt, key, _openrouter_call),
    "mistral": lambda http, prompt, key: call_baseline_openai_compat(http, prompt, key, _mistral_call),
}


# ── LLM-as-judge ─────────────────────────────────────────────


JUDGE_PROMPT = """\
You are a strict AI response evaluator. Rate the following response on a scale of 1-10.

Be discriminating — most responses should score between 5 and 8. Reserve 9-10 for exceptional quality.

Scoring rubric:
- Correctness (0-3): Is the information/code accurate? Would the code actually run? Any factual errors?
- Completeness (0-3): Does it address ALL parts of the prompt? Missing any requested element = max 1 here.
- Clarity (0-2): Is it well-structured? Easy to follow? Good formatting?
- Quality (0-2): Depth of analysis, code quality, writing sophistication. Generic/shallow = 0.

Evaluate each criterion, explain briefly (1-2 sentences per criterion), then give your total.
Output your final score on the last line in this exact format: [[score]]

Prompt: {prompt}

Response: {response}"""

PAIRWISE_PROMPT = """\
You are an expert AI response evaluator. Compare these two responses to the same prompt.

Prompt: {prompt}

--- Response A ---
{response_a}

--- Response B ---
{response_b}

Which response is better? Consider correctness, completeness, clarity, and quality.
Briefly explain your reasoning (2-3 sentences), then output your verdict on the last line:
- [[A]] if Response A is better
- [[B]] if Response B is better
- [[tie]] if they are roughly equal"""


# Judge provider config — uses Cerebras (not Groq) to avoid self-bias
# since Groq is a common baseline.
_JUDGE_PROVIDERS = [
    ("CEREBRAS_API_KEY", "https://api.cerebras.ai/v1/chat/completions", "qwen-3-235b-a22b-instruct-2507"),
    ("GEMINI_API_KEY", "", ""),  # special: Gemini uses different format
    ("GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile"),
]


def _find_judge_provider() -> tuple[str, str, str, str]:
    """Find the best available judge provider (prefer one NOT used as baseline)."""
    import os

    for env_var, url, model in _JUDGE_PROVIDERS:
        key = os.environ.get(env_var, "")
        if key:
            return env_var, url, model, key
    return "", "", "", ""


async def _call_judge_llm(http: httpx.AsyncClient, text: str, url: str, model: str, api_key: str) -> str:
    """Call the judge LLM and return raw text response."""
    resp = await http.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0.0,
            "max_tokens": 500,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _extract_score(text: str) -> float:
    """Extract [[score]] from judge response."""
    import re

    match = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", text)
    if match:
        return min(float(match.group(1)), 10.0)
    # Fallback: try to find a bare number at the end
    match = re.search(r"(\d+(?:\.\d+)?)\s*$", text)
    return min(float(match.group(1)), 10.0) if match else 0.0


def _extract_pairwise(text: str) -> str:
    """Extract [[A]], [[B]], or [[tie]] from judge response."""
    import re

    match = re.search(r"\[\[(A|B|tie)\]\]", text, re.IGNORECASE)
    return match.group(1).upper() if match else "TIE"


async def judge_response(
    http: httpx.AsyncClient,
    prompt: str,
    response: str,
    api_key: str,
    judge_url: str = "",
    judge_model: str = "",
) -> float:
    """Use LLM-as-judge to score a response (1-10) with chain-of-thought."""
    if not judge_url or not judge_model:
        return 0.0

    judge_text = JUDGE_PROMPT.replace("{prompt}", prompt).replace("{response}", response[:2000])

    try:
        raw = await _call_judge_llm(http, judge_text, judge_url, judge_model, api_key)
        return _extract_score(raw)
    except Exception:
        return 0.0


async def judge_pairwise(
    http: httpx.AsyncClient,
    prompt: str,
    response_a: str,
    response_b: str,
    api_key: str,
    judge_url: str = "",
    judge_model: str = "",
) -> str:
    """Pairwise comparison: returns 'A', 'B', or 'TIE'."""
    if not judge_url or not judge_model:
        return "TIE"

    text = (
        PAIRWISE_PROMPT.replace("{prompt}", prompt)
        .replace("{response_a}", response_a[:1500])
        .replace("{response_b}", response_b[:1500])
    )

    try:
        raw = await _call_judge_llm(http, text, judge_url, judge_model, api_key)
        return _extract_pairwise(raw)
    except Exception:
        return "TIE"


# ── Main benchmark runner ─────────────────────────────────────


async def run_benchmark(
    baselines: list[str],
    use_judge: bool = False,
    judge_key: str = "",
    proxy_url: str = DEFAULT_PROXY_URL,
) -> BenchmarkReport:
    """Run the full benchmark suite."""
    import os
    from datetime import datetime

    dataset = json.loads(DATASET_PATH.read_text())
    methods = ["smartsplit"] + baselines

    # Collect API keys for baselines
    api_keys: dict[str, str] = {}
    for name in baselines:
        env_var, _ = BASELINE_FACTORIES[name]
        key = os.environ.get(env_var, "")
        if not key:
            print(f"  WARNING: {env_var} not set, skipping {name} baseline")
            methods.remove(name)
        else:
            api_keys[name] = key

    # Find judge provider (prefer one not used as baseline to avoid self-bias)
    judge_env, judge_url, judge_model, judge_key_auto = _find_judge_provider()
    if not judge_key:
        judge_key = judge_key_auto
    if judge_env:
        print(f"  Judge: {judge_env} ({judge_model})")
    judge_url = judge_url or ""
    judge_model = judge_model or ""

    all_results: list[BenchmarkResult] = []

    async with httpx.AsyncClient() as http:
        # Check proxy is running
        try:
            health = await http.get(f"{proxy_url}/health", timeout=5.0)
            health.raise_for_status()
            print(f"  Proxy OK: {health.json()}")
        except Exception:
            print(f"  ERROR: SmartSplit proxy not running at {proxy_url}")
            print("  Start it with: smartsplit")
            return BenchmarkReport(
                timestamp=datetime.now(UTC).isoformat(),
                total_prompts=0,
                methods=methods,
            )

        total = len(dataset) * len(methods)
        done = 0

        for entry in dataset:
            prompt_id = entry["id"]
            category = entry["category"]
            prompt = entry["prompt"]
            entry_messages = entry.get("messages")

            for method in methods:
                done += 1
                print(f"  [{done}/{total}] {prompt_id} -> {method}...", end=" ", flush=True)

                start = time.monotonic()
                try:
                    if method == "smartsplit":
                        content, tokens = await call_smartsplit(http, prompt, proxy_url, messages=entry_messages)
                    else:
                        caller = BASELINE_CALLERS[method]
                        content, tokens = await caller(http, prompt, api_keys[method])

                    latency = int((time.monotonic() - start) * 1000)

                    result = BenchmarkResult(
                        prompt_id=prompt_id,
                        category=category,
                        method=method,
                        response=content,
                        latency_ms=latency,
                        estimated_tokens=tokens,
                        success=True,
                    )
                    print(f"OK ({latency}ms, ~{tokens} tokens)")

                except Exception as e:
                    latency = int((time.monotonic() - start) * 1000)
                    result = BenchmarkResult(
                        prompt_id=prompt_id,
                        category=category,
                        method=method,
                        response="",
                        latency_ms=latency,
                        estimated_tokens=0,
                        success=False,
                        error=str(e)[:200],
                    )
                    print(f"FAIL ({type(e).__name__})")

                all_results.append(result)

                # Rate limit protection — generous delay to avoid cross-contamination
                # between SmartSplit (which uses providers internally) and direct baselines.
                # Free tiers have strict rpm limits (Gemini: 15rpm, Cerebras: 30rpm).
                await asyncio.sleep(4.0)

        # Judge phase — absolute scoring
        if use_judge and judge_key and judge_url:
            print(f"\n  Scoring {len(all_results)} responses...")
            for i, result in enumerate(all_results):
                if result.success and result.response:
                    prompt = next(e["prompt"] for e in dataset if e["id"] == result.prompt_id)
                    result.judge_score = await judge_response(
                        http, prompt, result.response, judge_key,
                        judge_url=judge_url, judge_model=judge_model,
                    )
                    print(f"  [{i + 1}/{len(all_results)}] {result.prompt_id}/{result.method}: {result.judge_score}/10")
                    await asyncio.sleep(2.0)  # rate limit — judge provider has limits too

    # Build report
    report = BenchmarkReport(
        timestamp=datetime.now(UTC).isoformat(),
        total_prompts=len(dataset),
        methods=methods,
        results=[
            {
                "prompt_id": r.prompt_id,
                "category": r.category,
                "method": r.method,
                "latency_ms": r.latency_ms,
                "estimated_tokens": r.estimated_tokens,
                "success": r.success,
                "judge_score": r.judge_score,
                "error": r.error,
                "response_preview": r.response[:200] if r.response else "",
            }
            for r in all_results
        ],
        summary=_build_summary(all_results, methods, dataset),
    )

    return report


def _build_summary(
    results: list[BenchmarkResult],
    methods: list[str],
    dataset: list[dict],
) -> dict:
    """Compute aggregate stats per method, category, and domain."""
    categories = sorted(set(e["category"] for e in dataset))

    # Build domain lookup: prompt_id -> primary domain
    domain_lookup: dict[str, str] = {}
    for entry in dataset:
        domains = entry.get("expected_domains", ["general"])
        domain_lookup[entry["id"]] = domains[0] if domains else "general"

    summary: dict = {"by_method": {}, "by_category": {}, "by_domain": {}}

    for method in methods:
        method_results = [r for r in results if r.method == method]
        successes = [r for r in method_results if r.success]

        summary["by_method"][method] = {
            "total": len(method_results),
            "success_count": len(successes),
            "success_rate": round(len(successes) / max(len(method_results), 1) * 100, 1),
            "avg_latency_ms": round(sum(r.latency_ms for r in successes) / max(len(successes), 1)),
            "avg_tokens": round(sum(r.estimated_tokens for r in successes) / max(len(successes), 1)),
            "avg_judge_score": round(
                sum(r.judge_score for r in successes if r.judge_score > 0)
                / max(sum(1 for r in successes if r.judge_score > 0), 1),
                2,
            ),
        }

    for cat in categories:
        summary["by_category"][cat] = {}
        for method in methods:
            cat_results = [r for r in results if r.category == cat and r.method == method and r.success]
            summary["by_category"][cat][method] = {
                "success_count": len(cat_results),
                "avg_latency_ms": round(sum(r.latency_ms for r in cat_results) / max(len(cat_results), 1)),
                "avg_judge_score": round(
                    sum(r.judge_score for r in cat_results if r.judge_score > 0)
                    / max(sum(1 for r in cat_results if r.judge_score > 0), 1),
                    2,
                ),
            }

    # By domain (code, reasoning, translation, etc.)
    all_domains = sorted(set(domain_lookup.values()))
    for domain in all_domains:
        domain_prompt_ids = {pid for pid, d in domain_lookup.items() if d == domain}
        summary["by_domain"][domain] = {}
        for method in methods:
            domain_results = [
                r for r in results if r.prompt_id in domain_prompt_ids and r.method == method and r.success
            ]
            scored = [r for r in domain_results if r.judge_score > 0]
            summary["by_domain"][domain][method] = {
                "count": len(domain_results),
                "avg_judge_score": round(
                    sum(r.judge_score for r in scored) / max(len(scored), 1),
                    2,
                ),
            }

    # Win rates (SmartSplit vs each baseline on judge scores)
    if "smartsplit" in methods:
        win_rates: dict[str, dict] = {}
        for baseline in methods:
            if baseline == "smartsplit":
                continue
            wins = draws = losses = 0
            for entry in dataset:
                ss = next(
                    (
                        r
                        for r in results
                        if r.prompt_id == entry["id"] and r.method == "smartsplit" and r.judge_score > 0
                    ),
                    None,
                )
                bl = next(
                    (r for r in results if r.prompt_id == entry["id"] and r.method == baseline and r.judge_score > 0),
                    None,
                )
                if ss and bl:
                    if ss.judge_score > bl.judge_score:
                        wins += 1
                    elif ss.judge_score < bl.judge_score:
                        losses += 1
                    else:
                        draws += 1
            total = wins + draws + losses
            win_rates[f"smartsplit_vs_{baseline}"] = {
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_rate": round(wins / max(total, 1) * 100, 1),
            }
        summary["win_rates"] = win_rates

    return summary


def print_report(report: BenchmarkReport) -> None:
    """Pretty-print the benchmark results."""
    print("\n" + "=" * 70)
    print("  SMARTSPLIT BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Date: {report.timestamp}")
    print(f"  Prompts: {report.total_prompts}")
    print(f"  Methods: {', '.join(report.methods)}")

    if not report.summary:
        print("\n  No results.")
        return

    # By method
    print(f"\n{'Method':<15} {'Success':>8} {'Latency':>10} {'Tokens':>8} {'Score':>7}")
    print("-" * 50)
    for method, stats in report.summary["by_method"].items():
        print(
            f"  {method:<13} {stats['success_rate']:>6.1f}% "
            f"{stats['avg_latency_ms']:>7}ms "
            f"{stats['avg_tokens']:>6} "
            f"{stats['avg_judge_score']:>6.1f}"
        )

    # By category
    print("\n  By Category:")
    for cat, methods_stats in report.summary.get("by_category", {}).items():
        print(f"\n  {cat}:")
        for method, stats in methods_stats.items():
            score_str = f"{stats['avg_judge_score']:.1f}" if stats["avg_judge_score"] else "n/a"
            print(f"    {method:<13} {stats['avg_latency_ms']:>7}ms  score={score_str}")

    # By domain
    if "by_domain" in report.summary and any(
        any(s.get("avg_judge_score", 0) > 0 for s in d.values())
        for d in report.summary["by_domain"].values()
    ):
        print("\n  By Domain (judge scores):")
        header = f"  {'Domain':<14}"
        for method in report.methods:
            header += f" {method:>12}"
        print(header)
        print("  " + "-" * (14 + 13 * len(report.methods)))
        for domain, methods_stats in report.summary["by_domain"].items():
            row = f"  {domain:<14}"
            for method in report.methods:
                score = methods_stats.get(method, {}).get("avg_judge_score", 0)
                row += f" {score:>11.1f}" if score else f" {'n/a':>11}"
            print(row)

    # Win rates
    if "win_rates" in report.summary:
        print("\n  Win Rates (SmartSplit vs baselines):")
        for matchup, wr in report.summary["win_rates"].items():
            print(f"    {matchup}: {wr['wins']}W / {wr['draws']}D / {wr['losses']}L ({wr['win_rate']:.1f}%)")

    print("\n" + "=" * 70)


# ── CLI ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartSplit Benchmark")
    parser.add_argument(
        "--baselines", type=str, default="groq,gemini", help="Comma-separated baseline providers (groq,gemini,cerebras)"
    )
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-judge scoring")
    parser.add_argument("--proxy-url", type=str, default=DEFAULT_PROXY_URL)
    parser.add_argument("--output", type=str, default="", help="Output JSON file path")
    args = parser.parse_args()

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]

    print("\n  SmartSplit Benchmark")
    print(f"  Proxy: {args.proxy_url}")
    print(f"  Baselines: {baselines}")
    print(f"  Judge: {'yes' if args.judge else 'no'}")
    print()

    report = asyncio.run(run_benchmark(baselines, use_judge=args.judge, proxy_url=args.proxy_url))

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(RESULTS_DIR / f"benchmark_{report.timestamp[:10]}.json")
    Path(output_path).write_text(
        json.dumps(
            {
                "timestamp": report.timestamp,
                "total_prompts": report.total_prompts,
                "methods": report.methods,
                "results": report.results,
                "summary": report.summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\n  Results saved to: {output_path}")

    print_report(report)


if __name__ == "__main__":
    main()
