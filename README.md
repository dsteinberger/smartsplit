<div align="center">

# SmartSplit

**Free AI coding assistant. Multiple LLMs. Zero cost.**

[![CI](https://github.com/dsteinberger/smartsplit/actions/workflows/ci.yml/badge.svg)](https://github.com/dsteinberger/smartsplit/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)

One endpoint, multiple free LLMs.<br>
SmartSplit routes each request to the **best model for the task** — code to DeepSeek, search to Serper, translation to Mistral, reasoning to Gemini.

Works with **Continue** · **Cline** · **Aider** · **Tabby** · any OpenAI-compatible client.

[Quick Start](#quick-start) · [How It Works](#how-it-works) · [Providers](#providers) · [Metrics](#metrics)

</div>

---

## Why SmartSplit?

Free LLMs are good — but each one is good at **different things**:

| Task | Best free model | Why not the others? |
|------|----------------|-------------------|
| Code | **DeepSeek** (9/10) | Groq is fast but weaker at code |
| Reasoning | **Gemini** (8/10) | Cerebras is faster but less accurate |
| Translation | **Mistral** (9/10) | Others are mediocre at languages |
| Web search | **Serper** | LLMs can't search the web |
| Summaries | **Cerebras** (9/10) | Fastest inference, great quality |

Today, your coding assistant uses **one model for everything**. SmartSplit uses **the right model for each task**.

```
Your coding assistant (Continue, Cline, Aider...)
         |
    SmartSplit (localhost:8420)
         |
    ┌────┼──────────────────────────┐
    |    |          |                |
   Code  Search   Translate       Summary
    |    |          |                |
 DeepSeek Serper  Mistral        Cerebras
  (best)  (free)  (best)         (fastest)
    |    |          |                |
    └────┼──────────┼────────────────┘
         |
    Combined response
```

**No API subscription. No credit card. Just free API keys.**

### SmartSplit vs. single-model setup

| | Single model | SmartSplit |
|---|:---:|:---:|
| Code quality | One model does everything | **Best model per task** |
| Web search | Not available | **Built-in (Serper)** |
| Translation | Generic | **Mistral (specialist)** |
| Provider down? | You're stuck | **Auto-fallback** |
| Cost | $0-20/month | **$0** |

---

## Quick Start

### 1. Install

<table>
<tr><td><b>pip</b></td><td>

```bash
pip install smartsplit
```

</td></tr>
<tr><td><b>Docker</b></td><td>

```bash
docker run -p 8420:8420 -e GROQ_API_KEY=gsk_... smartsplit
```

</td></tr>
<tr><td><b>uv</b></td><td>

```bash
uv pip install smartsplit
```

</td></tr>
</table>

### 2. Get free API keys

You need **one key** to start (2 minutes):

| Provider | What it does | Free tier | Sign up |
|----------|-------------|-----------|---------|
| **Groq** | Fast LLM (Llama 3.3) | 14,400 req/day | [groq.com](https://groq.com) |
| **Serper** | Web search | 2,500 req/month | [serper.dev](https://serper.dev) |

> SmartSplit routes to the best model **among your configured providers**. With just Groq, everything goes to Groq. Add DeepSeek for better code, Mistral for translation, Gemini for reasoning.

### 3. Start SmartSplit

```bash
export GROQ_API_KEY="gsk_..."
export SERPER_API_KEY="..."

smartsplit
```

```
  SmartSplit — Free multi-LLM backend
  http://127.0.0.1:8420/v1
  Mode: balanced
```

### 4. Point your client

<details>
<summary><b>Continue</b> (config.yaml)</summary>

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
<summary><b>Cline</b></summary>

```bash
cline auth -p openai -k free -b http://localhost:8420/v1
```
</details>

<details>
<summary><b>Aider</b></summary>

```bash
aider --openai-api-base http://localhost:8420/v1 --openai-api-key free
```
</details>

<details>
<summary><b>Any OpenAI SDK</b></summary>

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8420/v1", api_key="free")
```
</details>

Done. Ask anything — SmartSplit picks the best free model automatically.

<details>
<summary><b>Other compatible clients</b></summary>

SmartSplit works with **any OpenAI-compatible client**:

| Client | Type | Setup |
|--------|------|-------|
| **Continue** | VS Code / JetBrains | `apiBase: http://localhost:8420/v1` |
| **Cline** | VS Code (agent) | `cline auth -b http://localhost:8420/v1` |
| **Aider** | Terminal | `--openai-api-base http://localhost:8420/v1` |
| **Cursor** | IDE | Custom endpoint in settings |
| **Open WebUI** | Web UI (self-hosted) | Add OpenAI connection |
| **Chatbox** | Desktop app | Custom API endpoint |
| **LibreChat** | Web UI (self-hosted) | OpenAI-compatible endpoint |
| **Jan** | Desktop app | Custom endpoint |
| **Tabby** | Code completion | Custom endpoint |
| **OpenAI SDK** | Python / JS / etc. | `base_url="http://localhost:8420/v1"` |

Any tool that lets you configure an OpenAI `apiBase` / `base_url` will work.

</details>

---

## How It Works

Every request is automatically classified into one of two modes:

### RESPOND — route to best model

Your prompt is decomposed into subtasks, each routed to the best free LLM:

```
"Write a Python function to parse CSV and handle errors"

  [code]       → DeepSeek  (score 9/10)
  [reasoning]  → Gemini    (score 8/10)
  [synthesis]  → Cerebras  (combines results)

  → Complete response in ~2 seconds
```

### ENRICH — web search first, then route

When the prompt needs current data, SmartSplit searches the web for free:

```
"What are the new features in Python 3.13?"

  [web_search] → Serper    (free Google results)
  [summarize]  → Cerebras  (formats the answer)

  → Response with real, current data
```

---

## Providers

### Routing table

```
Task          Best free providers (ranked)
─────────────────────────────────────────────
code          DeepSeek > OpenRouter > Gemini > Groq
reasoning     Cerebras > Gemini > DeepSeek > Groq
summarize     Cerebras > Groq = Gemini
translation   Mistral > Gemini > Groq
web search    Serper or Tavily
boilerplate   Cerebras > Groq > Gemini
```

### All supported providers

| Provider | Model | Free tier | Best at |
|----------|-------|-----------|---------|
| **Groq** | Llama 3.3 70B | 14,400 req/day | Fast, reliable |
| **Cerebras** | Llama 3.3 70B | Free tier | Ultra-fast |
| **Gemini** | 2.5 Flash | 500 req/day | Reasoning |
| **DeepSeek** | V3.2 | 500M tokens/month | Code |
| **Mistral** | Small | Free tier | Translation |
| **OpenRouter** | Qwen3 Coder 480B | 50 req/day | Code |
| **Serper** | Google Search | 2,500/month | Web search |
| **Tavily** | AI Search | 1,000/month | Web search |

**Add providers** by setting environment variables:

```bash
export GROQ_API_KEY="gsk_..."
export GEMINI_API_KEY="AIza..."
export DEEPSEEK_API_KEY="sk-..."
export CEREBRAS_API_KEY="csk-..."
export SERPER_API_KEY="..."
```

More providers = better routing, more fallbacks, higher resilience.

> SmartSplit also supports paid providers (Anthropic, OpenAI) as optional fallbacks. They're disabled by default — SmartSplit is 100% free out of the box.

---

## Reliability

| Feature | What it does |
|---------|-------------|
| **Circuit breaker** | 3 failures → provider auto-disabled for 30 min |
| **Quality gates** | Refusal detection ("I cannot...") → auto-escalation |
| **Fallback chains** | If one provider fails, the next takes over |
| **Decompose cache** | Repeated prompts skip analysis (LRU, 24h TTL) |

---

## Metrics

```bash
curl http://localhost:8420/metrics
```

```json
{
  "requests": { "total": 142, "enrich": 42, "respond": 100 },
  "savings": { "tokens_saved": 45000, "cost_saved_usd": 0.135 },
  "cache": { "hits": 23, "hit_rate": 16.2 },
  "circuit_breaker": { "unhealthy_providers": [] }
}
```

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
smartsplit --log-level DEBUG        # verbose logging
```
</details>

<details>
<summary><b>Config file</b> (alternative to env vars)</summary>

```bash
cp smartsplit.example.json smartsplit.json
# Edit with your API keys
```
</details>

<details>
<summary><b>Docker</b></summary>

```bash
docker build -t smartsplit .

docker run -p 8420:8420 \
  -e GROQ_API_KEY=gsk_... \
  -e SERPER_API_KEY=... \
  smartsplit

# With custom mode
docker run -p 8420:8420 \
  -e GROQ_API_KEY=gsk_... \
  smartsplit --mode economy
```
</details>

---

## Development

```bash
git clone https://github.com/dsteinberger/smartsplit.git
cd smartsplit
uv venv && uv pip install -e ".[dev]"

pytest tests/ -v          # 215 tests
ruff check smartsplit/    # lint
ruff format smartsplit/   # format
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Architecture

```
smartsplit/
  proxy.py           HTTP server + 2-mode triage + CLI
  formats.py         OpenAI format conversion + SSE streaming
  planner.py         Prompt decomposition + synthesis + LRU cache
  router.py          Provider scoring + routing + quality gates
  quota.py           Usage tracking + savings report
  config.py          Configuration + env vars
  models.py          Pydantic models + StrEnum
  providers/         One file per provider (3 lines for OpenAI-compatible)
```

Adding a new provider is **3 lines**:

```python
class NewProvider(OpenAICompatibleProvider):
    name = "new"
    api_url = "https://api.new.com/v1/chat/completions"
    model = "new-model"
```

---

<div align="center">

MIT License · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Changelog](CHANGELOG.md)

If SmartSplit saves you money, **[give it a star](https://github.com/dsteinberger/smartsplit)**.

</div>
