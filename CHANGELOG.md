# Changelog

All notable changes to SmartSplit will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### ⚠️ Breaking Changes

- **Architecture pivot: Proxy Anticipateur.** SmartSplit is no longer a multi-LLM router that decomposes prompts. It is now an OpenAI-compatible proxy in front of a single brain LLM, with two modes:
  - **Agent mode** (request has `tools`) — predict read-only tool calls and return FAKE `tool_use` responses to skip a brain round-trip.
  - **API mode** (no `tools`) — triage `TRANSPARENT` (forward as-is) or `ENRICH` (workers prep context, brain synthesises).
- **Package reorganisation.** Internal modules were moved into sub-packages: `smartsplit.tools.*`, `smartsplit.routing.*`, `smartsplit.triage.*`, `smartsplit.proxy.*`. Callers importing from `smartsplit.proxy`, `smartsplit.router`, `smartsplit.detector`, etc., must update their imports.
- **`/v1/chat/completions` no longer decomposes prompts into subtasks.** Requests are now forwarded to the brain (with optional enrichment). The old decomposition pipeline is gone.

### Added

- **HTTPS proxy mode** (`smartsplit --proxy`) — transparent HTTPS interception for Claude Code, Cline, Continue, Aider, OpenCode. Uses the client's own subscription auth, no SmartSplit API key needed.
- **`setup-claude` subcommand** — generates the CA certificate and prints launch instructions for Claude Code.
- **`/v1/messages` endpoint** — native Anthropic Messages API format (in addition to the existing OpenAI-compatible `/v1/chat/completions`).
- **Tool anticipation** — predicts read-only tool calls (rules → learned patterns → free LLM) and returns FAKE `tool_use` when confidence ≥ 0.85, saving a brain round-trip.
- **Tool pattern learner** — observes actual tool calls, learns patterns across 5 categories (file mentions, sequential, error keywords, project first-read, per-tool accuracy) with Wilson-score scoring and staleness decay. Persisted at `~/.smartsplit/tool_patterns.json`.
- **Fast triage detector** — keyword heuristic (<1 ms) decides `TRANSPARENT` vs `ENRICH` on most requests, with an LLM fallback (`detect_with_llm`) for ambiguous cases.
- **Multilingual keyword detection** — domain, intent, comparison, and analysis keywords available in 9 languages (EN, FR, ES, PT, DE, ZH, JA, KO, RU). Generated via `scripts/generate_i18n.py`.
- **4 enrichment types** routed to workers: `web_search`, `pre_analysis`, `multi_perspective`, `context_summary`.
- **Provider overrides** per task type: `{"overrides": {"code": "anthropic"}}`.
- **Brain auto-detection** from configured API keys (paid first, then free LLMs).
- **Perplexity provider** (Sonar / Sonar Pro) — paid, search-augmented.

### Changed

- **Circuit breaker hardened** — 5 weighted failure points within 2 minutes to open, exponential backoff (60 s → 30 min), 3 consecutive successes to fully reset. 401/403 trip immediately; 429 has half weight.
- **Per-call timeout** — 30 s cap on every provider call to prevent hanging upstreams from blocking routing.
- **Quality gate tuned** — error-pattern scan shortened to 80 chars, repetition threshold raised; low-quality responses fall through to the next provider instead of being returned.
- **MAB (UCB1) scoring** is now thread-safe with validated persistence.
- **Provider starvation fixed** — paid strong models get `cost_score = 0.2` (was 0.0) so they are not auto-excluded in economy mode.

### Fixed

- Single-provider setups (paid-only) now work: `call_free_llm()` falls back to the brain.
- `build_enriched_messages()` deep-copies the conversation so the fallback path keeps access to the originals.
- Race condition in `QuotaTracker` concurrent writes.

---

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
