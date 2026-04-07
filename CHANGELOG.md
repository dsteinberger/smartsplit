# Changelog

All notable changes to SmartSplit will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-04-07

### Added

- OpenAI-compatible HTTP endpoint (`/v1/chat/completions`)
- Automatic triage: RESPOND (route to best free LLM) and ENRICH (web search + route)
- Prompt decomposition into subtasks with domain detection
- Intelligent routing: scores each provider per task type (code, reasoning, translation, etc.)
- 8 free LLM providers: Groq, Cerebras, Gemini, DeepSeek, OpenRouter, Mistral, Anthropic, OpenAI
- 2 free search providers: Serper, Tavily
- Circuit breaker per provider (3 failures → 30 min cooldown)
- Quality gates with refusal detection and auto-escalation
- Decompose cache (LRU, 24h TTL)
- Metrics endpoint (`/metrics`) with triage stats, savings, cache hit rate
- Health endpoint (`/health`) and savings report (`/savings`)
- Docker support with multi-stage build
- CLI with mode selection (`--mode economy|balanced|quality`)
- Configuration via environment variables or JSON config file
