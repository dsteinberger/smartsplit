# SmartSplit — Project Guide

## What is this project?

SmartSplit is a **proxy anticipateur** — an OpenAI-compatible endpoint that makes your main LLM faster and smarter. It predicts what your LLM will need (files, searches, context) and pre-fetches it, and enriches complex requests with web search and pre-analysis. Works with Continue, Cline, Aider, OpenCode, or any OpenAI-compatible client.

## Architecture

SmartSplit operates in two modes based on the incoming request:

**Mode Agent** (request contains `tools`) — predict tool calls and save round-trips:
```
Client (with tools) → IntentionDetector predicts tool calls (free LLM)
                     → High confidence (≥0.85): FAKE tool_use (skip brain)
                     → Enrichment: web search, pre-analysis (if triggered)
                     → Request proxied to brain WITH tools preserved
                     → Agent loop works through SmartSplit (tool passthrough)
```

**Mode API** (no tools) — keyword + LLM triage (TRANSPARENT vs ENRICH):
```
Client (no tools) → Detector: TRANSPARENT or ENRICH
                   → TRANSPARENT: forward to brain, zero overhead
                   → ENRICH: workers search web, pre-analyze, compare
                   → Brain synthesizes with enriched context
```

```
smartsplit/
  cli.py                  # CLI entry point — argument parsing, mode dispatch
  proxy.py                # Single-process HTTPS proxy — TLS interception, dynamic certs, CONNECT tunneling
  pipeline.py             # Starlette app + SmartSplit pipeline (shared between proxy and API mode)
  intercept.py            # Shared interception logic — compression, prediction, fake response builders
  tool_registry.py        # Single source of truth for all tool definitions, aliases, categories, regex
  intention_detector.py   # Predicts read-only tool calls via rules + free LLM (SAFE_TOOLS filter, 0.85 threshold)
  tool_anticipator.py     # Executes anticipated tools locally (used in API mode only, not proxy mode)
  tool_pattern_learner.py # Learns from actual tool calls (Wilson score, 5 pattern types, staleness decay)
  anticipation.py         # Tool anticipation helpers — predict, filter, extract context
  detector.py             # Request triage — TRANSPARENT or ENRICH decision (keywords + LLM)
  enrichment.py           # ENRICH path — workers do prep work (web search, analysis), brain synthesizes
  formats.py              # OpenAI/Anthropic format conversion, SSE streaming, fake tool responses
  planner.py              # Domain detection (keywords + LLM), prompt decomposition, enrichment subtasks
  i18n_keywords.py        # Multilingual keywords (generated) — merged into planner + detector at load time
  router.py               # Worker scoring (Quality + Cost + Availability, additive weighted), routing + quality gates
  learning.py             # MAB (UCB1) adaptive scoring — auto-calibrates from real results
  quota.py                # Usage tracking, availability scores, savings report
  config.py               # Pydantic config, brain auto-detection, env var loading, defaults
  models.py               # Pydantic models + StrEnum (TaskType, Mode, Subtask, RouteResult…)
  exceptions.py           # SmartSplitError hierarchy
  mitm_addon.py           # mitmproxy addon (legacy) — standalone alternative to proxy.py
  providers/
    base.py              # ABC: LLMProvider, SearchProvider (Strategy pattern)
    registry.py          # Factory + lookup + call_brain + call_free_llm + circuit breaker
    groq.py / gemini.py / deepseek.py / mistral.py / cerebras.py / openrouter.py
    huggingface.py / cloudflare.py  # Free backup providers
    anthropic.py / openai.py   # Paid providers (optional)
    serper.py / tavily.py      # Search providers
scripts/
  generate_i18n.py   # One-shot script to generate/update i18n_keywords.py via deep-translator
```

## Key design decisions

- **Dual-mode proxy** — Agent mode (tools present) anticipates read-only tool calls; API mode (no tools) enriches with web search and pre-analysis
- **SAFE_TOOLS only** — NEVER execute write tools. Only read_file, Read, grep, Grep, web_search, WebSearch, list_files, Glob, list_directory, ls, find, cat, head, tail
- **Tool passthrough** — tools from the client are forwarded to the brain intact, agent loop works through SmartSplit
- **FAKE tool_use** — high confidence (≥0.85) predictions returned as fake tool responses, agent executes tools itself (saves 1 LLM round-trip)
- **Context injection** — enrichment results injected in last user message (cache-friendly, avoids prompt cache invalidation)
- **Conservative confidence** — 0.85 threshold for FAKE tool_use to avoid sending wrong predictions
- **Graceful degradation** — if anything in the anticipation pipeline fails, request falls through as transparent proxy
- **ToolPatternLearner** — observes actual tool calls, builds patterns (Wilson score, 5 types), improves predictions over time
- **Detector heuristic** (API mode) — fast (<1ms) TRANSPARENT/ENRICH decision without LLM call. Multilingual keywords (9 languages). LLM fallback for ambiguous cases.
- **TRANSPARENT path** — most API requests forward directly to brain, zero overhead
- **ENRICH path** — workers search web, pre-analyze, compare options → brain gets enriched context
- **Brain stays consistent** — same LLM for the entire session (compatible with agentic tools)
- **NEVER modify the brain's response** — forward it as-is to the client
- **Circuit breaker** per provider — 5 fails in 2 min → provider skipped with exponential backoff (30s → 60s → … up to 30 min). Success resets backoff.
- **Strategy pattern** for providers — add a new one in 3 lines
- **Pydantic models everywhere** — zero raw dicts, zero magic strings

## How to verify changes

```bash
make check    # lint + format check + tests
```
All tests must pass. No API key needed — tests are fully mocked.

## Code conventions

- Python 3.11+, `from __future__ import annotations` in every module
- Type hints on all function signatures
- Linted with ruff (`ruff check smartsplit/ tests/`)
- No `# type: ignore`, no `Any`
- Exceptions: use the custom hierarchy in `exceptions.py`
- Logging: `logging` module with lazy %-style formatting — `logger.info("text %s", var)`, never `logger.info(f"text {var}")`
- Never use `str.format()` with user input — concatenate instead

## When adding a language

1. Run `python scripts/generate_i18n.py --add <code>` (e.g. `--add hi` for Hindi)
2. Review the generated translations in `smartsplit/i18n_keywords.py`
3. Fix any bad translations (technical terms often need manual correction)
4. Add test cases in `tests/test_planner.py` (`test_multilingual_domain_detection`)
5. Add at least one enrich case in `tests/test_proxy.py` (`_ENRICH_CASES`)
6. Run `make check`

The i18n system works by merging translated keywords into the existing English keyword lists at module load time. No runtime translation, no extra dependency. English keywords live in `planner.py` and `proxy.py`; translations live in `i18n_keywords.py`.

Currently supported: EN, FR, ES, PT, DE, ZH, JA, KO, RU.

## When adding a new safe tool

All tool definitions live in `smartsplit/tool_registry.py` — the **single source of truth**. Never add tool names, file path regex, or well-known file lists directly in other files — import from `tool_registry.py`.

1. Add the canonical handler name to `CANONICAL_HANDLERS` in `tool_registry.py`
2. Add aliases (agent-specific names) to `TOOL_ALIAS` in `tool_registry.py`
3. Add to the right compression category: `DUMB_TOOLS` (small results), `SMART_TOOLS` (large, compress), or leave uncategorized (passthrough)
4. Add to the right type set if applicable: `READ_TOOLS`, `GREP_TOOLS`, `LIST_DIR_TOOLS`, `SEARCH_TOOLS`
5. Add an execution handler in `tool_anticipator.py` (in `_execute_one`, match on the canonical name)
6. Respect the security rules: read-only, sandboxed to working_dir, 5-second timeout, catch all errors
7. Add tests in `tests/test_tool_anticipator.py` — cover success, timeout, path traversal rejection
8. Run `make check` — the consistency tests in `tests/test_tool_registry.py` will catch any misalignment

## When adding a new provider

1. Create `smartsplit/providers/<name>.py` inheriting `LLMProvider` or `SearchProvider`
2. Add entry to `_PROVIDER_CLASSES` in `registry.py`
3. Add default config in `DEFAULT_PROVIDERS` in `config.py`
4. Add env var mapping in `_ENV_KEY_MAP` in `config.py`
5. Add scores in `DEFAULT_COMPETENCE_TABLE` in `config.py`
6. Add to `DEFAULT_FREE_LLM_PRIORITY` in `config.py` if it's a free LLM
7. Write tests, run the suite

## Don't

- Don't execute write tools — only SAFE_TOOLS (read_file, grep, web_search, list_directory and aliases)
- Don't modify the brain's response — forward it as-is to the client
- Don't remove tools from the request — tool passthrough is critical for agent loops
- Don't lower the confidence threshold below 0.7 — context pollution degrades quality fast
- Don't add providers without updating the competence table
- Don't use `str.format()` with user-provided prompts
- Don't use f-string in logger calls — use `logger.info("text %s", var)` for lazy evaluation
- Don't define tool names, `WELL_KNOWN_FILES`, or `FILE_REF_RE` outside `tool_registry.py` — import them
- Don't hardcode provider types (free/paid) — always read from config
- Don't expose API keys in logs or error messages
- Don't add keywords directly in `planner.py` or `proxy.py` for non-English — use `i18n_keywords.py`
- Don't edit `i18n_keywords.py` manually — use `scripts/generate_i18n.py` then review
