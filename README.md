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
All LLMs read files, search the web, and prepare context — while your main model focuses on thinking.
If any provider goes down, another steps in. No tokens wasted, no downtime.

Most providers offer free tiers — **start without spending a cent.** Add a paid key when you want more power.

[Why SmartSplit](#why-smartsplit) · [Quick Start](#quick-start) · [How It Works](#how-it-works) · [Providers](#providers)

</div>

---

## Why SmartSplit

### Agent mode: your rate limit is the real bottleneck

Every time your coding agent asks Claude to fix a bug, it starts a predictable dance:

```
You: "Fix the bug in auth.py"

  Claude → "I need to read auth.py"        → wait → gets the file
  Claude → "I need to read config.py too"  → wait → gets the file
  Claude → "Let me grep for the function"  → wait → gets results
  Claude → "Now I can fix it"              → writes the fix

  3 round-trips before writing a single line.
  Each one re-sends the FULL conversation (20-50K tokens).
```

**The real cost isn't money — it's your rate limit.** Anthropic tracks usage in sliding windows (5h/7d). Each round-trip eats into your budget. Hit 100% utilization → 429 → locked out until the window slides.

**SmartSplit intercepts the predictable reads.**

```
Same request, same tool — but through SmartSplit:

  SmartSplit sees "auth.py" in the prompt
    → predicts read_file(auth.py) + read_file(config.py)
    → returns FAKE tool_use instantly (0 tokens to Anthropic)
  Claude Code executes the reads locally, sends results back
  Claude gets the files in one shot → skips the back-and-forth

  3 round-trips saved = ~75K tokens not re-sent = more margin before 429.
```

### API mode: your LLM gets real-time data

When you use SmartSplit as an OpenAI-compatible endpoint (no agent tools), it enriches your prompts with data your LLM doesn't have — web search results, multi-perspective analysis — using free workers at zero cost.

No new tool to learn. Change one URL, that's it.

---

## Works with Claude Code

SmartSplit is a **native proxy for Claude Code**. Two modes, same result: fewer round-trips, fewer tokens.

### Proxy mode (recommended) — uses your subscription

```bash
smartsplit setup-claude                  # one-time setup (generates CA cert)

# Terminal 1
smartsplit --mode proxy

# Terminal 2
NODE_EXTRA_CA_CERTS=~/.smartsplit/certs/ca-cert.pem \
HTTPS_PROXY=http://localhost:8420 \
claude
```

This is the **recommended mode** for Claude Code. SmartSplit runs as a lightweight single-process HTTPS proxy — it intercepts requests to `api.anthropic.com` (and other LLM APIs), predicts tool calls, and returns instant FAKE responses when confident. Your **existing subscription auth** is forwarded untouched. No API key needed.

The proxy auto-generates TLS certificates (stored in `~/.smartsplit/certs/`) and handles CONNECT tunneling. Non-LLM traffic (GitHub, npm, etc.) passes through untouched via blind TCP relay.

### API mode — lightweight alternative

```bash
smartsplit                                        # start the proxy
ANTHROPIC_BASE_URL=http://localhost:8420 claude    # launch Claude Code through it
```

Simpler setup, but uses `ANTHROPIC_BASE_URL` which may have stricter rate limits on some accounts.

### What happens under the hood

- You say "fix the bug in auth.py"
- SmartSplit predicts Claude will call `read_file(auth.py)` and `read_file(test_auth.py)`
- High confidence → returns a **FAKE tool_use** response instantly (brain never called, 0 tokens)
- Claude Code executes the reads on your machine, sends results back
- Claude gets the files in one shot → **2 round-trips saved** → rate limit utilization stays lower
- If the enriched request triggers a 429, SmartSplit automatically retries with the original body

> **Note:** In Claude Code mode, the `SMARTSPLIT_BRAIN` setting is bypassed. SmartSplit always forwards to Anthropic using your Claude Code subscription token. Free workers (Groq, Cerebras, Gemini) are only used for the anticipation engine — they predict tool calls and pre-fetch files, they never replace Claude.

---

## Works with every agent

SmartSplit speaks **OpenAI** and **Anthropic** format natively. Any agent that lets you change the API endpoint works.

| Agent | Protocol | Setup |
|-------|----------|-------|
| **Claude Code** | HTTPS proxy | `smartsplit --mode proxy` + `NODE_EXTRA_CA_CERTS=~/.smartsplit/certs/ca-cert.pem HTTPS_PROXY=http://localhost:8420 claude` |
| **Claude Code** | Anthropic native | `ANTHROPIC_BASE_URL=http://localhost:8420 claude` |
| **Cline** | OpenAI compatible | Base URL: `http://localhost:8420/v1` |
| **Aider** | OpenAI / HTTPS proxy | `--openai-api-base http://localhost:8420/v1` or `HTTPS_PROXY` |
| **OpenCode** | OpenAI / HTTPS proxy | Config: `http://localhost:8420/v1` or `HTTPS_PROXY` |
| **Continue** | OpenAI compatible | `apiBase: http://localhost:8420/v1` |
| **Cursor** | OpenAI compatible | Custom API endpoint |
| **Any OpenAI client** | OpenAI compatible | `base_url="http://localhost:8420/v1"` |

The **HTTPS proxy mode** (`--mode proxy`) works with any agent that makes HTTPS requests — set `HTTPS_PROXY` and `NODE_EXTRA_CA_CERTS=~/.smartsplit/certs/ca-cert.pem` and SmartSplit intercepts transparently. The **API mode** (default) works with any OpenAI-compatible client.

For agents using the OpenAI endpoint, SmartSplit picks the best available LLM as brain (Claude > GPT > DeepSeek > Groq). Configure with `SMARTSPLIT_BRAIN` or let it auto-detect from your API keys.

---

## What SmartSplit does

### Agent mode — anticipates tool calls

When the request contains tools (Claude Code, Cline, Aider...), SmartSplit **predicts** what the LLM will read and pre-fetches it.

| | |
|---|---|
| **Saves rate limit budget** | Each FAKE tool_use = ~25K tokens not re-sent = slower utilization growth |
| **Fake tool_use** | High confidence (≥0.85)? Returns predicted tool calls instantly — 0 tokens to Anthropic |
| **5-10s faster** | FAKE responds in <1ms. Without it, Claude thinks 5-10s per tool call |
| **Adds web search** | Your agent doesn't have web tools? SmartSplit adds them for free via Serper |
| **Learns your workflow** | After a few sessions, it knows your project patterns — no LLM call needed |
| **Invisible** | All tools preserved, agent loop intact. SmartSplit is transparent to the LLM |

### API mode — enriches context

When the request has no tools (scripts, apps, simple API calls), SmartSplit **enriches** the context with data your LLM doesn't have.

| | |
|---|---|
| **Web search** | "Latest React features?" → real data from the web, not stale training |
| **Pre-analysis** | Complex prompt? Workers break it down so the brain gets structured context |
| **Comparison** | "Redis vs Memcached?" → multiple perspectives, not a one-sided answer |
| **Transparent** | Simple questions pass straight through. Zero overhead when not needed |

### Tool-Aware Proxy — classifies tool results

SmartSplit classifies tools into three categories for safe handling:

| Category | Tools | Behavior |
|----------|-------|----------|
| **Simple** | Read, list_directory, cat, Grep, git_status, git_show | Pass through as-is |
| **Smart** | WebSearch, WebFetch, git_log, git_diff, git_blame | Pass through (compression available but off by default) |
| **Decisional** | Write, Edit, Bash, send_message, create_pr | Never touch (brain decides content) |

This classification ensures SmartSplit only intercepts **read-only** tools and never touches write operations or execution results.

### The engine behind both modes

- **Rate limit protection** — each FAKE tool_use = ~25K tokens not re-sent to Anthropic = more margin before 429. Real-time monitoring of unified rate limit utilization (5h/7d sliding windows)
- **7 free LLMs** as workers — Groq, Cerebras, Gemini, OpenRouter, Mistral, HuggingFace, Cloudflare do the prep at zero cost
- **Auto-fallback** — provider down or rate-limited? Next one takes over in milliseconds
- **Smart 429 handling** — enriched request rate-limited? Retry with original body. Upstream 429 propagated to client (no retry storms)
- **~5-10s faster per round-trip** — FAKE tool_use responds in <1ms instead of waiting for Claude to think
- **9 languages** — English, French, Spanish, Portuguese, German, Chinese, Japanese, Korean, Russian
- **Never modifies the response** — what the brain says goes to you untouched

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

# Proxy mode (for Claude Code)
docker compose --profile proxy up -d proxy
```

> **Proxy mode** uses `Dockerfile.proxy` — certs are stored in a Docker volume (`certs`). Copy the CA cert from the volume or run `setup-claude` locally.

</details>

### 4. Connect your coding tool

<details open>
<summary><b>Claude Code</b> (recommended)</summary>

**Option A: Proxy mode** (recommended — full subscription quotas):

```bash
smartsplit setup-claude                  # one-time: generate CA cert

smartsplit --mode proxy                  # Terminal 1
NODE_EXTRA_CA_CERTS=~/.smartsplit/certs/ca-cert.pem \
HTTPS_PROXY=http://localhost:8420 \
claude                                   # Terminal 2
```

**Option B: API mode** (simpler setup):

```bash
smartsplit                               # Terminal 1
ANTHROPIC_BASE_URL=http://localhost:8420 claude  # Terminal 2
```

Both options use your existing Claude subscription. SmartSplit anticipates tool calls, compresses tool results, and Claude does the rest. No API key needed.

> **Note:** Proxy mode requires `cryptography` (included by default). Docker works fine — SmartSplit predicts tool calls and the agent executes them on your machine.

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

**That's it.** Three steps: install, add one API key, connect your tool. Your assistant now has access to every top free LLM.

---

## How It Works

### Agent mode — under the hood

```
Your request: "Fix the import error in proxy.py" (with tools)

  1. SmartSplit intercepts the request (HTTPS CONNECT → TLS termination)
  2. Rule-based check: "proxy.py" mentioned → predict read_file(proxy.py) (0ms, regex)
  3. LLM predictor (free, ~200ms): "the LLM will also grep for imports"
  4. High-confidence predictions (≥0.85) → FAKE tool_use response (skip brain entirely)
     The agent executes the tools itself, sends results back → brain gets context in one shot
  5. Detector checks for enrichment: web question? → workers search + pre-analyze
  6. Large tool results compressed before reaching brain (Tool-Aware Proxy)
  7. Request proxied to brain — with ALL tools preserved
  8. If upstream returns 429 on enriched request → auto-retry with original body
  9. ToolPatternLearner records what the LLM actually called → better next time
```

Three layers of prediction, from fastest to smartest:

| Layer | Speed | How | Example |
|-------|-------|-----|---------|
| **Rules** | 0ms | Regex: file mentioned in prompt → read it | "fix auth.py" → `read_file(auth.py)` |
| **Patterns** | 0ms | Learned from past sessions (Wilson score, staleness decay) | "after pytest fails → read the failing test" |
| **LLM** | ~200ms | Free LLM predicts tool calls | Understands intent, predicts grep/search |

Confidence threshold controls what happens with predictions:

| Confidence | Action |
|-----------|--------|
| **≥ 0.85** | **FAKE tool_use** — return predicted tool calls instantly, agent executes them itself |
| **< 0.85** | **Skipped** — too risky, 10% irrelevant content degrades quality by ~23% |

### API mode — under the hood

```
Your request: "What's new in Python 3.13? Write code using the best feature."

  1. Fast detector (<1ms): keywords + prompt length → needs web search
  2. LLM triage (if ambiguous): confirms enrichment type
  3. Worker (Serper): searches the web → finds Python 3.13 release notes
  4. Worker (Groq): pre-analyzes the request structure
  5. Brain receives: original prompt + real web data + analysis
  6. Brain answers with current, factual information
```

| Signal | Action |
|--------|--------|
| Request has `tools` | **Agent mode**: anticipate read-only tool calls |
| Web/current data ("latest", "2026", "search"...) | **Enrich**: web search |
| Complex prompt (> 200 chars, multi-domain) | **Enrich**: pre-analysis |
| Comparison ("vs", "pros and cons"...) | **Enrich**: multi-perspective |
| Long conversation (> 10 messages) | **Enrich**: context summary |
| Simple question, code task | **Transparent**: zero overhead |

### Brain auto-detection

SmartSplit picks the best available LLM as the brain (priority: paid first, then free by capability):

| Keys configured | Brain | Workers |
|----------------|-------|---------|
| + Anthropic key | Claude | All free providers |
| + OpenAI key | GPT-4o | All free providers |
| + DeepSeek key | DeepSeek | All free providers |
| Only free keys | Groq (LLaMA 3.3 70B) | Gemini, Cerebras, Mistral... |

The brain stays **consistent** across the session — compatible with agentic tools (OpenCode, Aider, Cline). Override with `SMARTSPLIT_BRAIN=groq`.

### Built-in reliability

| Feature | What it does |
|---------|-------------|
| **Circuit breaker** | 5 failures in 2 min → exponential backoff (30s → 60s → ... → 30 min) |
| **429 auto-retry** | Enriched request rate-limited? Auto-retry with original body on a fresh connection |
| **Stale detection** | Files recently written by the LLM are skipped (avoids serving outdated content) |
| **Graceful degradation** | Anticipation pipeline fails → request falls through as transparent proxy |
| **Brain fallback** | Brain down → next best provider takes over |
| **Quality gates** | Detects refusals ("I cannot...") → auto-escalation |
| **Worker isolation** | Worker failure → skip, never blocks the response |
| **Context preservation** | Full conversation history passed to brain |
| **Adaptive scoring** | Learns which workers perform best (MAB/UCB1) |

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

> **Format translation is automatic.** Most providers use the OpenAI format natively. Gemini uses Google's own format — SmartSplit translates on the fly. Your client talks OpenAI, SmartSplit handles the rest.

> **Paid providers** (Anthropic, OpenAI) are also supported as optional fallbacks. They're disabled by default.

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
smartsplit                          # defaults: port 8420, balanced mode
smartsplit --port 3456              # custom port
smartsplit --mode economy           # max free usage
smartsplit --mode quality           # prefer quality over speed
smartsplit --mode proxy             # HTTPS proxy (lightweight, single-process, built-in TLS)
smartsplit --mode proxy-mitm        # HTTPS proxy via mitmproxy (legacy, heavier)
smartsplit setup-claude             # one-time setup helper (generates CA cert)
smartsplit --log-level DEBUG        # verbose logging
```
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

# Proxy mode (for Claude Code)
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

## Architecture

SmartSplit operates in two modes based on the incoming request:

**Agent mode** (request contains `tools`) — predict tool calls and save round-trips:
```
Client (with tools) → IntentionDetector predicts tool calls (free LLM)
                    → High confidence (≥0.85): FAKE tool_use response (skip brain)
                    → Enrichment: web search, pre-analysis (if triggered)
                    → Large tool results compressed (Tool-Aware Proxy)
                    → Request proxied to brain WITH tools preserved
                    → Agent loop works through SmartSplit (tool passthrough)
```

**API mode** (no tools) — keyword + LLM triage:
```
Client (no tools) → Detector: TRANSPARENT or ENRICH
                  → TRANSPARENT: forward to brain, zero overhead
                  → ENRICH: workers search web, pre-analyze, compare
                  → Brain synthesizes with enriched context
```

```
smartsplit/
  cli.py                  CLI entry point — argument parsing, mode dispatch
  proxy.py                Single-process HTTPS proxy — TLS interception, dynamic certs, CONNECT tunneling
  pipeline.py             Starlette app + SmartSplit pipeline (API mode + shared Anthropic pipeline)
  intercept.py            Shared interception logic — compression, prediction, fake response builders
  anticipation.py         Agent mode orchestration (predict → pre-execute → inject)
  detector.py             Triage TRANSPARENT/ENRICH (API mode, keyword + LLM)
  enrichment.py           ENRICH path — workers search web, pre-analyze, brain synthesizes
  intention_detector.py   Predicts read-only tool calls (rules + patterns + LLM, 9 languages)
  tool_anticipator.py     Pre-executes anticipated tools (SAFE_TOOLS only, sandboxed, 5s timeout)
  tool_registry.py        Single source of truth for all tool definitions, aliases, categories
  tool_pattern_learner.py Learns from actual tool calls (Wilson score, 5 pattern types, staleness decay)
  formats.py              OpenAI + Anthropic format conversion, SSE streaming, fake tool responses
  planner.py              Domain detection, prompt decomposition, enrichment subtask generation
  router.py               Worker scoring + routing + quality gates
  learning.py             MAB (UCB1) adaptive scoring — learns from real results
  quota.py                Usage tracking + savings report
  config.py               Configuration, brain auto-detection, env vars
  models.py               Pydantic models + StrEnum
  exceptions.py           Custom error hierarchy
  i18n_keywords.py        Multilingual keywords (9 languages, generated via scripts/generate_i18n.py)
  mitm_addon.py           mitmproxy addon (legacy alternative to proxy.py)
  providers/              One file per provider (Strategy pattern)
```

Adding a new provider is **2 lines** (model is set in config):

```python
class NewProvider(OpenAICompatibleProvider):
    name = "new"
    api_url = "https://api.new.com/v1/chat/completions"
```

---

## Works great with

SmartSplit enriches the **inputs** to your LLM. These tools optimize the **outputs** — they're complementary.

| Tool | What it does | How it helps |
|------|-------------|--------------|
| [**RTK**](https://github.com/rtk-ai/rtk) | Filters tool outputs before they reach the LLM (Rust, 40+ commands) | 60-90% token reduction on git, tests, linters |
| [**Snip**](https://github.com/edouard-claude/snip) | Same concept, declarative YAML filters (Go) | Extensible, drop-in filter files |

SmartSplit + RTK = your LLM gets **clean inputs** (RTK removes noise) **and enriched context** (SmartSplit adds web search, pre-analysis). Both install in minutes, zero conflict.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `curl /health` returns nothing | SmartSplit isn't running | Check the terminal for errors. At least one API key is required |
| `No available provider` in logs | All providers are down or no key configured | Check `curl localhost:8420/health` — add more keys for fallback |
| `429 Too Many Requests` from Claude Code | Anthropic rate limit hit | Normal — SmartSplit auto-retries with the original body. Wait for the window to slide |
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
