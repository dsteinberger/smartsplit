<div align="center">

<img src="assets/banner.png" alt="SmartSplit — Why use one LLM when you can use them all?" width="100%">



> [!WARNING]
> This project is under active development and not yet stable. APIs may change without notice.

[![CI](https://github.com/dsteinberger/smartsplit/actions/workflows/ci.yml/badge.svg)](https://github.com/dsteinberger/smartsplit/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/codecov/c/github/dsteinberger/smartsplit?logo=codecov&logoColor=white)](https://codecov.io/gh/dsteinberger/smartsplit)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)

**Why use one LLM when you can use them all?**

Your coding agent works alone. SmartSplit gives it a team.<br>
Multiple LLMs search the web, analyze context, and anticipate what your main model will need — while it focuses on thinking.
If any provider goes down, another steps in. No tokens wasted, no downtime.

Most providers offer free tiers — **start without spending a cent.** Add a paid key when you want more power.

[What It Does](#what-smartsplit-does) · [Quick Start](#quick-start) · [Features](#features) · [Providers](#providers)

</div>

---

## What SmartSplit does

SmartSplit gives your main LLM an entire team. Connect any combination of LLMs — free or paid — and they work in parallel to read files, search the web, analyze context, and prepare everything your brain needs. Your main model receives pre-digested, enriched context and focuses on what it does best: thinking.

Each LLM has its strengths. SmartSplit knows them and routes the right task to the right worker — code analysis to one, web search to another, translation to a third. The more models you connect, the stronger the team: Gemini's reasoning combined with Groq's speed, Mistral's multilingual skills, or a paid model like GPT-4o for deeper analysis. If one goes down, the next takes over automatically.

On top of that, SmartSplit **saves round-trips** for coding agents. When your agent needs to fix a bug, the LLM typically makes 3-4 back-and-forth calls just to read the right files — each one re-sending the full conversation. SmartSplit predicts those reads and pre-fetches them, so the LLM gets everything in one shot.

The result: **faster responses, lower token usage, real-time data your LLM doesn't have — powered by the combined strengths of every model you connect.**

SmartSplit adapts automatically based on what your client sends:

| Your client sends | SmartSplit does |
|---|---|
| A request **with tools** (Claude Code, Cline, Aider...) | Workers predict which files the LLM will need — the agent pre-fetches them, saving round-trips |
| A request **without tools** (scripts, apps, simple API calls) | Workers search the web, pre-analyze, compare — brain gets enriched context |
| A simple question | Forwards it directly — zero overhead |

Both paths share the same engine: the same workers, the same fallback logic, the same learning. And the brain's response is **never modified** — what your LLM says goes to you untouched.

<details>
<summary><b>How does a coding agent actually work? (and why SmartSplit helps)</b></summary>

A coding agent (Claude Code, Cursor, Cline...) works like this:

```
1. You type: "Fix the bug in auth.py"

2. The agent sends your message to the LLM (Claude, GPT, Gemini...)
   along with a list of TOOLS the LLM can use:
   - Read(path)    — read a file
   - Write(path)   — write a file
   - Bash(command)  — run a command
   - Grep(pattern)  — search in files

3. The LLM responds: "I want to use Read(auth.py)" (a "tool call")

4. The agent executes Read(auth.py) on your machine
   and sends the file content back to the LLM

5. The LLM reads, thinks, and says: "Now I want Read(config.py)"
   → another round-trip

6. This loop continues until the LLM has enough context.
   Each round = a new HTTP request with the FULL conversation history.
```

**The cost:** each round-trip re-sends everything — system prompt, all messages, all previous tool results. A 10-turn task sends the same context 10 times. That's a lot of tokens.

**SmartSplit's insight:** most reads are predictable. "Fix auth.py" → the LLM will read `auth.py`. Pytest fails → it will read the failing test file. These patterns repeat across every project, every developer.

SmartSplit predicts them and pre-executes the reads **before** the LLM asks. The LLM still has all its tools — it can ask for anything. But the obvious reads are already in context, so it can skip straight to the actual work.

**Fewer round-trips = fewer tokens = lower cost = faster results.**

</details>

---

## Quick Start

### 1. Install

```bash
pip install smartsplit
# or: uv pip install smartsplit
```

### 2. Get a free API key (2 minutes)

You need **one key** to start. Sign up at [groq.com](https://groq.com) and copy your API key.

> Add more providers later for better routing. Each new provider = better results, more fallbacks. See [Providers](#providers).

### 3. Start SmartSplit

```bash
export GROQ_API_KEY="gsk_..."
smartsplit
```

```
  SmartSplit — Multi-LLM backend
  http://127.0.0.1:8420/v1
  Mode: balanced
```

Verify it's running:

```bash
curl http://localhost:8420/health
```

<details>
<summary><b>Or use Docker</b></summary>

```bash
# Create a .env from the template
cp .env.example .env   # then edit with your API keys
# or: make env

# Run with Docker (API mode)
docker run -p 8420:8420 --env-file .env ghcr.io/dsteinberger/smartsplit

# Or with Docker Compose
docker compose up -d

# Proxy mode (for HTTPS interception)
docker compose --profile proxy up -d proxy
```

> **Proxy mode** uses `Dockerfile.proxy` — certs are stored in a Docker volume (`certs`). Copy the CA cert from the volume or run `smartsplit setup-claude` locally.

</details>

### 4. Connect your coding tool

SmartSplit speaks **OpenAI**, **Anthropic**, and **Gemini** formats natively. Any agent that lets you change the API endpoint works.

<details open>
<summary><b>Claude Code</b></summary>

```bash
smartsplit setup-claude                  # one-time: generate CA cert

smartsplit --proxy                  # Terminal 1
NODE_EXTRA_CA_CERTS=~/.smartsplit/certs/ca-cert.pem \
HTTPS_PROXY=http://localhost:8420 \
claude                                   # Terminal 2
```

SmartSplit intercepts requests to `api.anthropic.com`, predicts tool calls, and returns instant responses when confident. Your subscription auth is forwarded untouched — no API key needed. Non-LLM traffic passes through via blind TCP relay.

</details>

<details>
<summary><b>Cline</b> (VS Code)</summary>

In the Cline sidebar, click the gear icon:
1. Select **OpenAI Compatible** as provider
2. Base URL: `http://localhost:8420/v1`
3. API Key: `free`
4. Model ID: `smartsplit`
</details>

<details>
<summary><b>Aider</b> (Terminal)</summary>

Copy [`examples/.aider.conf.yml`](examples/.aider.conf.yml) to your project as `.aider.conf.yml`, or run:

```bash
aider --model openai/smartsplit --openai-api-base http://localhost:8420/v1 --openai-api-key free
```
</details>

<details>
<summary><b>Continue</b> (VS Code / JetBrains)</summary>

Copy [`examples/.continuerc.json`](examples/.continuerc.json) to your project as `.continuerc.json`, or add to `~/.continue/config.yaml`:

```yaml
models:
  - name: SmartSplit
    provider: openai
    model: smartsplit
    apiBase: http://localhost:8420/v1
    apiKey: free
```
</details>

<details>
<summary><b>OpenCode</b> (Terminal)</summary>

Copy [`examples/opencode.json`](examples/opencode.json) to your project root, run `opencode providers` to add the API key (`free`), then select the model with `/models`.
</details>

<details>
<summary><b>Tabby</b> (Self-hosted autocomplete)</summary>

Add to `~/.tabby/config.toml`:

```toml
[model.chat.http]
kind = "openai/chat"
model_name = "smartsplit"
api_endpoint = "http://localhost:8420/v1"
api_key = "free"
```
</details>

<details>
<summary><b>Void</b> (Open-source IDE)</summary>

In Void settings:
1. Find **OpenAI-Compatible** section → set Base URL `http://localhost:8420/v1`, API Key `free`
2. In **Models** section → Add Model, select OpenAI-Compatible, name: `smartsplit`
</details>

<details>
<summary><b>Any OpenAI-compatible client</b></summary>

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8420/v1", api_key="free")
```

SmartSplit works with **any tool that supports a custom OpenAI endpoint**: Continue, Cline, Aider, OpenCode, Tabby, Void, Cursor, Open WebUI, Chatbox, LibreChat, Jan, and more.
</details>

**That's it.** Install, add one API key, connect your tool.

> **Which LLM is the brain?** In proxy mode, your agent's own LLM is the brain (e.g. Claude with your subscription). In API mode, SmartSplit auto-detects the best available brain from your API keys (Claude > GPT > DeepSeek > Groq). Override anytime with `SMARTSPLIT_BRAIN=groq`.
>
> **Proxy vs API mode:** Proxy mode (`--proxy`) intercepts HTTPS requests transparently — works with any agent that makes HTTPS calls. API mode (default) exposes an OpenAI-compatible endpoint. Both offer the same SmartSplit features.

---

## Features

### Core — shared by all modes

| Feature | What it does |
|---------|-------------|
| **Brain** | Your main LLM — auto-detected from your API keys (Claude, GPT, DeepSeek...) or set with `SMARTSPLIT_BRAIN`. Stays consistent across the session |
| **Workers** | All other connected LLMs do the prep work — read files, search the web, analyze context. The more you connect, the stronger the team |
| **Web search** | Workers search the web via Serper/Tavily — your brain gets real-time data it doesn't have |
| **Auto-fallback** | Provider down or rate-limited? Next one takes over in milliseconds |
| **Circuit breaker** | 5 failures in 2 min → exponential backoff (30s → 60s → ... → 30 min). Auto-recovery |
| **Smart 429 handling** | Enriched request rate-limited? Retry with original body. Upstream 429 propagated (no retry storms) |
| **Adaptive scoring** | Learns which workers perform best (MAB/UCB1) — routing improves over time |
| **Quality gates** | Detects refusals ("I cannot...") → auto-escalation to a better provider |
| **9 languages** | English, French, Spanish, Portuguese, German, Chinese, Japanese, Korean, Russian |
| **Never modifies the response** | What the brain says goes to you untouched |

### When your client sends tools (agent mode)

When the request contains tools, SmartSplit **predicts** which read-only tools the LLM will call and pre-fetches the results.

| Feature | What it does |
|---------|-------------|
| **FAKE tool_use** | Confidence ≥ 0.85 → returns predicted tool calls instantly (0 tokens to the brain, ~5-10s faster) |
| **3-layer prediction** | Rules (0ms, regex) → Learned patterns (0ms, Wilson score) → Free LLM (~200ms, intent analysis) |
| **Pattern learning** | Records what the LLM actually calls → predictions improve over time (5 pattern types, staleness decay) |
| **Tool-Aware Proxy** | Classifies tool results: simple (pass-through), smart (compressible), decisional (never touch) |
| **Tool passthrough** | All tools preserved — the agent loop works through SmartSplit, invisible to the LLM |
| **Rate limit protection** | Each FAKE tool_use = ~25K tokens not re-sent = more margin before 429 |

### When your client doesn't send tools (API mode)

When there are no tools, SmartSplit detects whether the request benefits from enrichment.

| Feature | What it does |
|---------|-------------|
| **Transparent pass-through** | Simple questions forward directly — zero overhead |
| **Web enrichment** | "Latest React features?" → real data from the web, not stale training |
| **Pre-analysis** | Complex prompt → workers break it down so the brain gets structured context |
| **Comparison** | "Redis vs Memcached?" → multiple perspectives, not a one-sided answer |

---

## How It Works

```
                    ┌─────────────────────────────────────┐
                    │          Your coding tool           │
                    │  (Claude Code, Cline, Aider, ...)   │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                         ┌─────────────────┐
                         │   SmartSplit     │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
             Has tools?                   No tools?
                    │                           │
                    ▼                           │
     ┌──────────────────────────┐              │
     │  IN PARALLEL:            │              │
     │                          │              │
     │  1. Predict tool calls   │              │
     │     (rules, patterns,    │              │
     │      free LLM)           │              │
     │                          │              │
     │  2. Detect enrichment    │              │
     │     (web search,         │              │
     │      pre-analysis)       │              │
     └─────────┬────────────────┘              │
               │                               │
               ▼                               ▼
      High confidence              ┌───────────────────┐
        (≥ 0.85)?                  │  Detect: needs    │
      ┌────┴────┐                  │  enrichment?      │
      │         │                  │  (keywords + LLM) │
    YES        NO                  └─────────┬─────────┘
      │         │                       ┌────┴────┐
      ▼         │                       │         │
 FAKE tool_use  │                    ENRICH   TRANSPARENT
 (0 tokens,     │                       │         │
  agent         │                       ▼         │
  executes      │                Workers search   │
  locally)      │                web, pre-analyze  │
      │         │                       │          │
      │         └───────┬───────────────┘          │
      │                 ▼                          │
      │     Forward to brain                       │
      │     (+ enrichment if available,            │
      │      tools preserved)                      │
      │                 │                          │
      └────────────┬────┘                          │
                   ▼                               │
           Brain responds ◄────────────────────────┘
                   │
                   ▼
          Response forwarded
          untouched to client
```

### Prediction layers (with tools)

| Layer | Speed | How | Example |
|-------|-------|-----|---------|
| **Rules** | 0ms | Regex: file mentioned in prompt → read it | "fix auth.py" → `read_file(auth.py)` |
| **Patterns** | 0ms | Learned from past sessions (Wilson score, staleness decay) | "after pytest fails → read the failing test" |
| **LLM** | ~200ms | Free LLM predicts tool calls | Understands intent, predicts grep/search |

### Enrichment detection

| Signal | Action |
|--------|--------|
| Web/current data ("latest", "2026", "search"...) | Web search |
| Complex prompt (> 200 chars, multi-domain) | Pre-analysis |
| Comparison ("vs", "pros and cons"...) | Multi-perspective |
| Long conversation (> 10 messages) | Context summary (API mode only) |
| Simple question, code task | Transparent — zero overhead |

### Brain auto-detection

SmartSplit picks the best available LLM as the brain:

| Keys configured | Brain | Workers |
|----------------|-------|---------|
| + Anthropic key | Claude | All free providers |
| + OpenAI key | GPT-4o | All free providers |
| + DeepSeek key | DeepSeek | All free providers |
| Only free keys | Groq (LLaMA 3.3 70B) | Gemini, Cerebras, Mistral... |

Override with `SMARTSPLIT_BRAIN=groq`. The brain stays consistent across the session — compatible with agentic tool loops.

---

## Providers

### Supported providers

| Provider | Type | Best at |
|----------|------|---------|
| **Cerebras** | Free | Reasoning, general (Qwen 3 235B A22B) |
| **Groq** | Free | Fast inference (LLaMA 3.3 70B) |
| **Gemini** | Free | Math, factual, reasoning (Gemini 2.5 Flash) |
| **OpenRouter** | Free | Code (Qwen3 Coder) |
| **Mistral** | Free | Translation (Mistral Small) |
| **HuggingFace** | Free backup | Code (Qwen2.5 Coder 32B) |
| **Cloudflare** | Free backup | General (LLaMA 3.3 70B) |
| **DeepSeek** | Paid | Code, reasoning |
| **Anthropic** | Paid | Complex tasks (Claude) |
| **OpenAI** | Paid | Complex tasks (o3) |
| **Serper** | Free | Web search |
| **Tavily** | Free | Web search |

Add providers by setting environment variables:

```bash
export GROQ_API_KEY="gsk_..."
export GEMINI_API_KEY="AIza..."
export DEEPSEEK_API_KEY="sk-..."
export CEREBRAS_API_KEY="csk-..."
export MISTRAL_API_KEY="..."
export OPENROUTER_API_KEY="sk-or-..."
export HF_TOKEN="hf_..."
export CLOUDFLARE_API_KEY="..."
export CLOUDFLARE_ACCOUNT_ID="..."
export SERPER_API_KEY="..."
export TAVILY_API_KEY="tvly-..."
```

More providers = better routing, more fallbacks, higher resilience.

> **Format translation is automatic.** Most providers use the OpenAI format natively. Gemini uses Google's own format — SmartSplit translates on the fly. Your client talks OpenAI or Anthropic, SmartSplit handles the rest.

> **Paid providers** (Anthropic, OpenAI) are also supported as optional brains or fallbacks. They're disabled by default.

<details>
<summary><b>Routing table</b></summary>

```
Task          Best free providers (ranked)
─────────────────────────────────────────────
code          OpenRouter > Cerebras = Gemini > Groq = HuggingFace
reasoning     Cerebras > Gemini = OpenRouter > Groq
summarize     Cerebras > Groq = Gemini = Mistral = OpenRouter
translation   Mistral > Gemini > Groq = Cerebras
web search    Serper or Tavily
boilerplate   Cerebras = Groq > Gemini = Mistral = OpenRouter
math          OpenRouter = Gemini > Cerebras > Groq
general       Cerebras > Gemini = OpenRouter > Groq = Mistral

Backups:      HuggingFace, Cloudflare (lower quality, high availability)
```
</details>

---

## Metrics

```bash
curl http://localhost:8420/metrics
```

```json
{
  "requests": { "total": 142, "transparent": 100, "enrich": 42 },
  "anticipation": {
    "requests_with_tools": 89,
    "predictions_made": 67,
    "tools_anticipated": 134,
    "tools_injected": 98,
    "files_already_read": 12,
    "pattern_learner": { "total_observations": 203 }
  },
  "savings": { "tokens_saved": 45000, "cost_saved_usd": 0.135 },
  "circuit_breaker": { "unhealthy_providers": [] }
}
```

The `anticipation` section tells you exactly what's working: how many predictions were made, how many tools were successfully pre-fetched, and how many were skipped (already read or recently written).

Also available: `GET /health` · `GET /savings`

---

## Configuration

<details>
<summary><b>CLI options</b></summary>

```bash
# API mode (default) — SmartSplit picks a brain and routes workers
smartsplit                          # defaults: port 8420, balanced mode
smartsplit --mode economy           # favor free providers, lower quality gates
smartsplit --mode balanced          # default — balance quality and cost
smartsplit --mode quality           # favor quality, stricter quality gates

# Proxy mode — intercepts HTTPS, your agent keeps its own LLM
smartsplit --proxy                  # HTTPS proxy (balanced routing by default)
smartsplit --proxy --mode quality   # HTTPS proxy with quality-first routing
smartsplit setup-claude             # one-time setup helper (generates CA cert)

# Common options
smartsplit --port 3456              # custom port
smartsplit --log-level DEBUG        # verbose logging
```

> **Note:** `economy`, `balanced`, and `quality` control how SmartSplit scores and selects workers for enrichment. This applies to both API and proxy modes. In proxy mode, your agent's own LLM stays the brain — only the worker routing is affected.
</details>

<details>
<summary><b>Config file</b> (alternative to env vars)</summary>

```bash
cp smartsplit.example.json smartsplit.json
# Edit with your API keys
```

You can also tune provider settings and routing:

```json
{
  "mode": "balanced",
  "free_llm_priority": ["cerebras", "groq", "gemini", "openrouter", "mistral", "huggingface", "cloudflare"],
  "providers": {
    "groq": {
      "model": "llama-3.3-70b-versatile",
      "temperature": 0.3,
      "max_tokens": 4096
    },
    "serper": {
      "max_search_results": 5
    }
  }
}
```

| Option | Default | What it does |
|--------|---------|-------------|
| `free_llm_priority` | cerebras, groq, gemini, openrouter, mistral, huggingface, cloudflare | Fallback order for free LLM calls |
| `providers.*.model` | per-provider default | Override the default model |
| `providers.*.temperature` | `0.3` | LLM temperature |
| `providers.*.max_tokens` | `4096` | Max output tokens |
| `providers.*.max_search_results` | `5` | Number of web search results |

</details>

<details>
<summary><b>Docker</b></summary>

```bash
# Create .env from template
cp .env.example .env   # then edit with your API keys

# API mode (default)
docker build -t smartsplit .
docker run -p 8420:8420 --env-file .env smartsplit

# Proxy mode
docker build -f Dockerfile.proxy -t smartsplit-proxy .
docker run -p 8420:8420 --env-file .env -v smartsplit-certs:/certs smartsplit-proxy
```

Or use Docker Compose:
```bash
docker compose up -d                          # API mode
docker compose --profile proxy up -d proxy    # proxy mode
```

Or use the published image:
```bash
docker run -p 8420:8420 --env-file .env ghcr.io/dsteinberger/smartsplit
```

> Never commit `.env` to git — it's already in `.gitignore`.

</details>

---

## Development

**Prerequisites:** Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip.

```bash
git clone https://github.com/dsteinberger/smartsplit.git
cd smartsplit
make install              # or: pip install -e ".[dev]"

make check                # lint + format check + tests
make test                 # tests only
make run                  # start server (requires at least one API key)
make proxy                # start in HTTPS proxy mode (lightweight, built-in TLS)
make setup-claude         # one-time Claude Code setup helper (generates CA cert)
make help                 # all commands
```

> **Note:** `make test` runs all tests without any API key — no provider needed for development.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `curl /health` returns nothing | SmartSplit isn't running | Check the terminal for errors. At least one API key is required |
| `No available provider` in logs | All providers are down or no key configured | Check `curl localhost:8420/health` — add more keys for fallback |
| `429 Too Many Requests` | Rate limit hit | Normal — SmartSplit auto-retries with the original body. Wait for the window to slide |
| Proxy mode: `CERTIFICATE_VERIFY_FAILED` | CA cert not trusted | Run `smartsplit setup-claude` and set `NODE_EXTRA_CA_CERTS=~/.smartsplit/certs/ca-cert.pem` |
| Proxy mode: requests hang | Port conflict or proxy not started | Verify `HTTPS_PROXY=http://localhost:8420` and that SmartSplit is running in proxy mode |
| `circuit breaker open` for a provider | 5 failures in 2 min → auto-skip | Provider recovers automatically (exponential backoff). Check `/metrics` for details |

---

## Disclaimer

SmartSplit is a personal development tool. Each user must provide their own API keys and comply with the terms of service of each provider they use. SmartSplit does not store, share, or redistribute API keys or access. The authors are not responsible for any misuse or ToS violations by end users.

---

<div align="center">

MIT License · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Changelog](CHANGELOG.md)

**[Star this repo](https://github.com/dsteinberger/smartsplit)** to follow updates — new providers, streaming, and more coming soon.

</div>
