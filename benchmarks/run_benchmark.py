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


BASELINE_FACTORIES = {
    "groq": ("GROQ_API_KEY", _groq_call),
    "gemini": ("GEMINI_API_KEY", _gemini_call),
    "cerebras": ("CEREBRAS_API_KEY", _cerebras_call),
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


BASELINE_CALLERS = {
    "groq": call_baseline_groq,
    "gemini": call_baseline_gemini,
    "cerebras": call_baseline_cerebras,
}


# ── LLM-as-judge ─────────────────────────────────────────────


JUDGE_PROMPT = """\
You are a strict evaluator. Rate the following AI response on a scale of 1-10.

Criteria:
- Completeness: Does the response address ALL parts of the prompt?
- Accuracy: Is the information correct?
- Coherence: Is the response well-structured and easy to follow?
- Quality: Is the code correct (if applicable)? Is the writing good?

Prompt: {prompt}

Response: {response}

Output ONLY a single integer from 1 to 10, nothing else."""


async def judge_response(
    http: httpx.AsyncClient,
    prompt: str,
    response: str,
    api_key: str,
) -> float:
    """Use Groq (free) as LLM-as-judge to score a response."""
    judge_text = JUDGE_PROMPT.replace("{prompt}", prompt).replace("{response}", response[:2000])

    try:
        resp = await http.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": judge_text}],
                "temperature": 0.0,
                "max_tokens": 5,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        score_text = resp.json()["choices"][0]["message"]["content"].strip()
        return float(score_text)
    except Exception:
        return 0.0


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

    if not judge_key:
        judge_key = os.environ.get("GROQ_API_KEY", "")

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

                # Rate limit protection — longer delay to avoid cross-contamination
                # between SmartSplit (which uses providers internally) and direct baselines
                await asyncio.sleep(1.5)

        # Judge phase
        if use_judge and judge_key:
            print(f"\n  Judging {len(all_results)} responses...")
            for i, result in enumerate(all_results):
                if result.success and result.response:
                    prompt = next(e["prompt"] for e in dataset if e["id"] == result.prompt_id)
                    result.judge_score = await judge_response(
                        http,
                        prompt,
                        result.response,
                        judge_key,
                    )
                    print(f"  [{i + 1}/{len(all_results)}] {result.prompt_id}/{result.method}: {result.judge_score}/10")
                    await asyncio.sleep(0.3)  # rate limit

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
    """Compute aggregate stats per method and category."""
    categories = sorted(set(e["category"] for e in dataset))

    summary: dict = {"by_method": {}, "by_category": {}}

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
