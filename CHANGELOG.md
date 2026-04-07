# Changelog

All notable changes to SmartSplit will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-04-07

### Added

- OpenAI-compatible HTTP endpoint (`/v1/chat/completions`)
- LLM-based triage: RESPOND (route to best LLM) and ENRICH (web search + route). Language-agnostic with keyword fallback
- Prompt decomposition into subtasks with domain detection
- Intelligent routing: scores each provider per task type (code, reasoning, translation, etc.)
- 8 free LLM providers: Groq, Cerebras, Gemini, OpenRouter, Mistral, HuggingFace, Cloudflare (+ DeepSeek paid)
- 2 optional paid LLM providers: Anthropic, OpenAI (disabled by default)
- 2 search providers: Serper, Tavily
- Real token usage tracking (aggregated across providers, returned in OpenAI format)
- Circuit breaker per provider (3 failures → 30 min cooldown)
- Quality gates with refusal detection and auto-escalation
- Decompose cache (LRU, 24h TTL)
- Fast/strong model tiers for paid providers (Haiku/Sonnet, GPT-4o-mini/GPT-4o)
- Automatic tier selection based on task complexity
- Full conversation context preservation (messages passed to LLMs, not just last prompt)
- Context summary injection for multi-subtask decomposition
- HTTP 503 with structured error when all providers are unavailable
- Metrics endpoint (`/metrics`) with triage stats, savings, cache hit rate
- Health endpoint (`/health`) and savings report (`/savings`)
- Docker support with lightweight image, HEALTHCHECK, and Docker Compose
- Docker image published to GitHub Container Registry on release
- CLI with mode selection (`--mode economy|balanced|quality`)
- Configuration via environment variables or JSON config file
- Daily Provider Watch job (automated LLM ecosystem monitoring via GitHub Actions)
- GitHub issue templates, PR template, Dependabot, pre-commit hooks
- Adaptive scoring via MAB (UCB1) — learns which providers work best from real usage
- Codecov integration for test coverage tracking
- Makefile with common commands (`make check`, `make run`, `make docker-build`, etc.)
- Example config files for OpenCode, Aider, Continue in `examples/`
