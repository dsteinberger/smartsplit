"""Benchmark: detector accuracy, enrichment value, and tool prediction.

Tests the 3 core pipelines of SmartSplit's anticipatory proxy:
  1. Detector — does heuristic + LLM triage correctly classify TRANSPARENT vs ENRICH?
  2. Enrichment — does web search + pre-analysis actually improve the brain's response?
  3. Tool prediction — does IntentionDetector predict the right read-only tools?

Usage:
    # Detector accuracy only (no API keys needed)
    python benchmarks/bench_synthesis.py --detector-only

    # Full benchmark (needs API keys)
    export GROQ_API_KEY="gsk_..."
    export CEREBRAS_API_KEY="csk_..."
    export GEMINI_API_KEY="..."
    export SERPER_API_KEY="..."
    python benchmarks/bench_synthesis.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

# ── Detector test cases ──────────────────────────────────────

DETECTOR_CASES: list[dict] = [
    # TRANSPARENT — simple code tasks, explanations, short prompts
    {"prompt": "Fix the typo in line 42", "expected": "transparent", "category": "code"},
    {"prompt": "What does this function do?", "expected": "transparent", "category": "code"},
    {"prompt": "Rename the variable foo to bar", "expected": "transparent", "category": "code"},
    {"prompt": "Add type hints to this module", "expected": "transparent", "category": "code"},
    {"prompt": "Write a Python function to sort a list", "expected": "transparent", "category": "code"},
    {"prompt": "Explain how async/await works in Python", "expected": "transparent", "category": "explanation"},
    {"prompt": "Why is my test failing with AssertionError?", "expected": "transparent", "category": "debug"},
    {"prompt": "Refactor this class to use dataclasses", "expected": "transparent", "category": "code"},
    {"prompt": "yes", "expected": "transparent", "category": "short"},
    {"prompt": "ok do it", "expected": "transparent", "category": "short"},
    {"prompt": "Can you add logging to the retry logic?", "expected": "transparent", "category": "code"},
    {"prompt": "Delete the unused imports", "expected": "transparent", "category": "code"},
    # ENRICH — web search, comparisons, current data
    {
        "prompt": "What are the best Python testing frameworks in 2026? Compare pytest, unittest, and hypothesis with pros and cons.",
        "expected": "enrich",
        "category": "comparison",
    },
    {
        "prompt": "Find the latest version of FastAPI and what changed in the most recent release.",
        "expected": "enrich",
        "category": "web_search",
    },
    {
        "prompt": "Compare Redis vs Memcached vs DragonflyDB for session caching — performance benchmarks, pros and cons, and which one to choose for a high-traffic Python web app.",
        "expected": "enrich",
        "category": "comparison",
    },
    {
        "prompt": "What are the current best practices for securing a FastAPI application? Include OWASP recommendations and recent CVEs.",
        "expected": "enrich",
        "category": "web_search",
    },
    {
        "prompt": "Is there a newer alternative to Celery for task queues in Python? Compare the options with benchmarks.",
        "expected": "enrich",
        "category": "comparison",
    },
    {
        "prompt": "What's the current status of PEP 750? Has it been accepted?",
        "expected": "enrich",
        "category": "web_search",
    },
]

# ── Tool prediction test cases ───────────────────────────────

PREDICTION_CASES: list[dict] = [
    {
        "id": "read_file",
        "messages": [
            {"role": "user", "content": "Can you look at the config file?"},
            {"role": "assistant", "content": "I'll read the config file for you."},
            {"role": "user", "content": "Check smartsplit/config.py"},
        ],
        "expected_tools": ["read_file"],
        "expected_args": {"path": "smartsplit/config.py"},
    },
    {
        "id": "grep_search",
        "messages": [
            {"role": "user", "content": "Where is SAFE_TOOLS defined?"},
        ],
        "expected_tools": ["grep"],
        "expected_args": {"pattern": "SAFE_TOOLS"},
    },
    {
        "id": "list_dir",
        "messages": [
            {"role": "user", "content": "What files are in the tests directory?"},
        ],
        "expected_tools": ["list_directory"],
        "expected_args": {"path": "tests"},
    },
    {
        "id": "multi_tool",
        "messages": [
            {"role": "user", "content": "Read the README and check what's in the smartsplit/ folder"},
        ],
        "expected_tools": ["read_file", "list_directory"],
    },
    {
        "id": "no_tool",
        "messages": [
            {"role": "user", "content": "Write a function that adds two numbers"},
        ],
        "expected_tools": [],
    },
]

# Available tools (simplified OpenAI format for the predictor)
AVAILABLE_TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "grep", "description": "Search file contents", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "list_directory", "description": "List directory contents", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write content to a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}},
]

# ── Enrichment A/B test prompts ──────────────────────────────

ENRICHMENT_PROMPTS = [
    {
        "id": "lib_comparison",
        "prompt": "Compare Pydantic v2 vs attrs vs dataclasses for a high-performance Python API. Which should I use and why?",
    },
    {
        "id": "current_info",
        "prompt": "What's the recommended way to deploy a Python web app on AWS Lambda in 2026?",
    },
    {
        "id": "security",
        "prompt": "What are the latest best practices for JWT token handling in Python APIs? Any recent vulnerabilities to watch for?",
    },
]


# ── Benchmark runners ────────────────────────────────────────


def run_detector_benchmark() -> dict:
    """Test detector accuracy — no API keys needed."""
    from smartsplit.triage.detector import detect

    results = {"correct": 0, "wrong": 0, "details": []}

    for case in DETECTOR_CASES:
        decision, enrichments = detect(case["prompt"])
        actual = decision.value
        expected = case["expected"]
        correct = actual == expected

        if correct:
            results["correct"] += 1
        else:
            results["wrong"] += 1

        results["details"].append({
            "prompt": case["prompt"][:60],
            "category": case["category"],
            "expected": expected,
            "actual": actual,
            "enrichments": enrichments,
            "correct": correct,
        })

    total = results["correct"] + results["wrong"]
    results["accuracy"] = results["correct"] / total if total > 0 else 0
    return results


async def run_prediction_benchmark(registry: object) -> dict:
    """Test tool prediction accuracy — needs free LLM API keys."""
    from smartsplit.tools.intention_detector import IntentionDetector
    from smartsplit.tools.pattern_learner import ToolPatternLearner

    pattern_learner = ToolPatternLearner(project_dir=".")
    detector = IntentionDetector(registry, pattern_learner=pattern_learner)

    results = {"correct": 0, "partial": 0, "wrong": 0, "details": []}

    for case in PREDICTION_CASES:
        case_id = case["id"]
        print(f"    [{case_id}]...", end=" ", flush=True)
        t0 = time.monotonic()

        try:
            prediction = await detector.predict(case["messages"], AVAILABLE_TOOLS)
            predicted_tools = [t.tool for t in prediction.tools if t.confidence >= 0.5]
        except Exception as e:
            print(f"FAIL: {e}")
            results["wrong"] += 1
            results["details"].append({"id": case_id, "error": str(e)})
            continue

        elapsed = time.monotonic() - t0
        expected = set(case["expected_tools"])
        actual = set(predicted_tools)

        if actual == expected:
            status = "EXACT"
            results["correct"] += 1
        elif expected and actual & expected:
            status = "PARTIAL"
            results["partial"] += 1
        elif not expected and not actual:
            status = "EXACT"
            results["correct"] += 1
        else:
            status = "WRONG"
            results["wrong"] += 1

        print(f"{status} ({elapsed:.1f}s) expected={sorted(expected)} got={sorted(actual)}")
        results["details"].append({
            "id": case_id,
            "expected": sorted(expected),
            "actual": sorted(actual),
            "confidence": [f"{t.tool}:{t.confidence:.2f}" for t in prediction.tools],
            "status": status,
            "elapsed": round(elapsed, 2),
        })

    total = results["correct"] + results["partial"] + results["wrong"]
    results["accuracy_exact"] = results["correct"] / total if total > 0 else 0
    results["accuracy_partial"] = (results["correct"] + results["partial"]) / total if total > 0 else 0
    return results


async def run_enrichment_benchmark(registry: object, http: object) -> dict:
    """A/B test: brain response with vs without enrichment.

    Sends the same prompt twice — once raw, once with enrichment context prepended.
    Uses a separate LLM as judge to compare quality.
    """
    from smartsplit.config import load_config
    from smartsplit.triage.enrichment import enrich_only
    from smartsplit.routing.learning import BanditScorer
    from smartsplit.proxy.pipeline import ProxyContext
    from smartsplit.triage.planner import Planner
    from smartsplit.routing.quota import QuotaTracker
    from smartsplit.routing.router import Router

    cfg = load_config()
    quota = QuotaTracker(provider_configs=cfg.providers)
    planner = Planner(registry)
    bandit = BanditScorer()
    router = Router(registry, quota, cfg, bandit=bandit)

    ctx = ProxyContext(
        config=cfg,
        registry=registry,
        planner=planner,
        router=router,
        quota=quota,
        bandit=bandit,
        http=http,
    )

    _DELAY = 3.0
    results = {"enriched_wins": 0, "raw_wins": 0, "ties": 0, "details": []}

    for test in ENRICHMENT_PROMPTS:
        prompt = test["prompt"]
        prompt_id = test["id"]
        print(f"    [{prompt_id}]")

        # Get raw response from free LLM
        print("      Raw...", end=" ", flush=True)
        t0 = time.monotonic()
        try:
            raw_response = await registry.call_free_llm(prompt)
            print(f"OK ({time.monotonic() - t0:.1f}s, {len(raw_response)} chars)")
        except Exception as e:
            print(f"FAIL: {e}")
            continue
        await asyncio.sleep(_DELAY)

        # Get enrichment context
        print("      Enrich...", end=" ", flush=True)
        t0 = time.monotonic()
        try:
            enrichment_results = await enrich_only(ctx, prompt, ["web_search", "pre_analysis"])
            context = "\n".join(f"[{r.type.value}]: {r.response}" for r in enrichment_results if r.response)
            print(f"OK ({time.monotonic() - t0:.1f}s, {len(context)} chars context)")
        except Exception as e:
            print(f"FAIL: {e}")
            continue
        await asyncio.sleep(_DELAY)

        # Get enriched response (context + prompt)
        print("      Enriched...", end=" ", flush=True)
        t0 = time.monotonic()
        enriched_prompt = f"Use this research context to inform your answer:\n\n{context}\n\n---\n\nUser question: {prompt}"
        try:
            enriched_response = await registry.call_free_llm(enriched_prompt)
            print(f"OK ({time.monotonic() - t0:.1f}s, {len(enriched_response)} chars)")
        except Exception as e:
            print(f"FAIL: {e}")
            continue
        await asyncio.sleep(_DELAY)

        # Judge
        print("      Judge...", end=" ", flush=True)
        judge_prompt = (
            "You are comparing two AI responses to the same question.\n\n"
            f"Question: {prompt}\n\n"
            f"--- Response A ---\n{raw_response[:3000]}\n\n"
            f"--- Response B ---\n{enriched_response[:3000]}\n\n"
            "Which response is more accurate, complete, and useful? "
            "Consider factual accuracy, specificity (named versions, real projects), and depth.\n"
            "Reply with ONLY: [[A]], [[B]], or [[tie]]"
        )
        try:
            import re

            verdict_raw = await registry.call_free_llm(judge_prompt)
            match = re.search(r"\[\[(A|B|tie)\]\]", verdict_raw, re.IGNORECASE)
            verdict = match.group(1).upper() if match else "TIE"
        except Exception:
            verdict = "TIE"

        if verdict == "B":
            results["enriched_wins"] += 1
            print("ENRICHED wins")
        elif verdict == "A":
            results["raw_wins"] += 1
            print("RAW wins")
        else:
            results["ties"] += 1
            print("TIE")

        results["details"].append({
            "id": prompt_id,
            "verdict": verdict,
            "raw_len": len(raw_response),
            "enriched_len": len(enriched_response),
            "context_len": len(context),
        })
        await asyncio.sleep(_DELAY)

    return results


# ── Main ─────────────────────────────────────────────────────


def print_detector_results(results: dict) -> None:
    accuracy = results["accuracy"]
    print(f"\n  {'Prompt':<62} {'Expected':<13} {'Actual':<13} {'':>3}")
    print("  " + "-" * 94)
    for d in results["details"]:
        mark = "OK" if d["correct"] else "XX"
        enr = f" ({', '.join(d['enrichments'])})" if d["enrichments"] else ""
        print(f"  {d['prompt']:<62} {d['expected']:<13} {d['actual'] + enr:<25} {mark}")
    print(f"\n  Accuracy: {results['correct']}/{results['correct'] + results['wrong']} ({accuracy:.0%})")
    if results["wrong"] > 0:
        wrong = [d for d in results["details"] if not d["correct"]]
        print(f"  Misclassified: {len(wrong)}")
        for d in wrong:
            print(f"    - [{d['category']}] {d['prompt']} (expected={d['expected']}, got={d['actual']})")


def print_prediction_results(results: dict) -> None:
    print(f"\n  Exact: {results['correct']}  Partial: {results['partial']}  Wrong: {results['wrong']}")
    print(f"  Accuracy (exact): {results['accuracy_exact']:.0%}")
    print(f"  Accuracy (partial+): {results['accuracy_partial']:.0%}")


def print_enrichment_results(results: dict) -> None:
    total = results["enriched_wins"] + results["raw_wins"] + results["ties"]
    print(f"\n  Enriched wins: {results['enriched_wins']}/{total}")
    print(f"  Raw wins: {results['raw_wins']}/{total}")
    print(f"  Ties: {results['ties']}/{total}")
    for d in results["details"]:
        print(f"    [{d['id']}] {d['verdict']} (raw={d['raw_len']}c, enriched={d['enriched_len']}c, context={d['context_len']}c)")


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SmartSplit benchmark suite")
    parser.add_argument("--detector-only", action="store_true", help="Run only detector accuracy test (no API keys)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  SmartSplit Benchmark Suite")
    print("=" * 60)

    # 1. Detector benchmark (always runs, no API keys needed)
    print("\n  [1/3] Detector Accuracy (heuristic)")
    print("  " + "-" * 40)
    detector_results = run_detector_benchmark()
    print_detector_results(detector_results)

    if args.detector_only:
        print("\n" + "=" * 60)
        return

    # Check API keys
    has_keys = bool(os.environ.get("GROQ_API_KEY") or os.environ.get("CEREBRAS_API_KEY"))
    if not has_keys:
        print("\n  Skipping prediction + enrichment benchmarks (no API keys).")
        print("  Set GROQ_API_KEY / CEREBRAS_API_KEY / SERPER_API_KEY to run full suite.")
        print("\n" + "=" * 60)
        return

    # Initialize registry
    import httpx

    from smartsplit.config import load_config
    from smartsplit.providers.registry import ProviderRegistry

    cfg = load_config()
    http_client = httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
    )
    registry = ProviderRegistry(
        cfg.providers, http_client, free_llm_priority=cfg.free_llm_priority, brain_name=cfg.brain,
    )

    # 2. Tool prediction benchmark
    print("\n  [2/3] Tool Prediction Accuracy")
    print("  " + "-" * 40)
    prediction_results = await run_prediction_benchmark(registry)
    print_prediction_results(prediction_results)

    # 3. Enrichment value benchmark
    has_search = bool(os.environ.get("SERPER_API_KEY") or os.environ.get("TAVILY_API_KEY"))
    if has_search:
        print("\n  [3/3] Enrichment Value (A/B test)")
        print("  " + "-" * 40)
        enrichment_results = await run_enrichment_benchmark(registry, http_client)
        print_enrichment_results(enrichment_results)
    else:
        print("\n  [3/3] Skipped (no SERPER_API_KEY or TAVILY_API_KEY)")

    await http_client.aclose()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
