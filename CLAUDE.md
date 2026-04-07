# SmartSplit — Project Guide

## What is this project?

SmartSplit is a **free multi-LLM backend** — an OpenAI-compatible endpoint that routes each request to the best free LLM for the task. Code goes to DeepSeek, search to Serper, translation to Mistral, reasoning to Gemini. Works with Continue, Cline, Aider, or any OpenAI-compatible client.

Target: developers without paid API subscriptions who want an intelligent AI coding assistant for free.

## Architecture

```
smartsplit/
  proxy.py           # HTTP server: Starlette app, LLM-based triage (RESPOND/ENRICH), CLI
  formats.py         # OpenAI/Anthropic request/response conversion + SSE streaming
  planner.py         # Decomposes prompts via free LLM, synthesizes results, LRU cache
  router.py          # Scores providers (Quality × Availability × Budget), routes subtasks
  learning.py        # MAB (UCB1) adaptive scoring — auto-calibrates from real results
  quota.py           # Usage tracking, availability scores, savings report
  config.py          # Pydantic config, env var loading, defaults
  models.py          # Pydantic models + StrEnum (TaskType, Mode, Subtask, RouteResult…)
  exceptions.py      # SmartSplitError hierarchy
  providers/
    base.py          # ABC: LLMProvider, SearchProvider (Strategy pattern)
    registry.py      # Factory + lookup + fallback + circuit breaker
    groq.py / gemini.py / deepseek.py / mistral.py / cerebras.py / openrouter.py
    huggingface.py / cloudflare.py  # Free backup providers
    anthropic.py / openai.py   # Paid providers (optional)
    serper.py / tavily.py      # Search providers
```

## Key design decisions

- **Free-first backend** — not a proxy, SmartSplit IS the LLM backend for the client
- **LLM-based triage** — RESPOND (route to best LLM) or ENRICH (web search first, then respond). Uses LLM classification with keyword fallback for language-agnostic domain detection
- **Decomposition** — multi-domain prompts are split into subtasks, each routed independently
- **Circuit breaker** per provider — 3 fails in 5 min → provider skipped for 30 min
- **Decompose cache** — LRU + TTL cache skips planner on repeated prompts
- **Strategy pattern** for providers — add a new one in 3 lines
- **Pydantic models everywhere** — zero raw dicts, zero magic strings

## How to verify changes

```bash
source .venv/bin/activate && python -m pytest tests/ -v
```
All tests must pass.

```bash
timeout 3 python -m smartsplit 2>&1 || true
```

## Code conventions

- Python 3.11+, `from __future__ import annotations` in every module
- Type hints on all function signatures
- Linted with ruff (`ruff check smartsplit/ tests/`)
- No `# type: ignore`, no `Any`
- Exceptions: use the custom hierarchy in `exceptions.py`
- Logging: `logging` module throughout
- Never use `str.format()` with user input — concatenate instead

## When adding a new provider

1. Create `smartsplit/providers/<name>.py` inheriting `LLMProvider` or `SearchProvider`
2. Add entry to `_PROVIDER_CLASSES` in `registry.py`
3. Add default config in `DEFAULT_PROVIDERS` in `config.py`
4. Add env var mapping in `_ENV_KEY_MAP` in `config.py`
5. Add scores in `DEFAULT_COMPETENCE_TABLE` in `config.py`
6. Add to `DEFAULT_FREE_LLM_PRIORITY` in `config.py` if it's a free LLM
7. Write tests, run the suite

## Don't

- Don't add providers without updating the competence table
- Don't use `str.format()` with user-provided prompts
- Don't hardcode provider types (free/paid) — always read from config
- Don't expose API keys in logs or error messages
