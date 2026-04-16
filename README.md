<div align="center">

<img src="assets/banner.png" alt="SmartSplit — Why use one LLM when you can use them all?" width="100%">



[![CI](https://github.com/dsteinberger/smartsplit/actions/workflows/ci.yml/badge.svg)](https://github.com/dsteinberger/smartsplit/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/codecov/c/github/dsteinberger/smartsplit?logo=codecov&logoColor=white)](https://codecov.io/gh/dsteinberger/smartsplit)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)

**Why use one LLM when you can use them all?**

Smart routing to the best model for each task — picks the right model tier automatically.<br>
Free by default. Optimizes paid tokens when available.

[Quick Start](#quick-start) · [How It Works](#how-it-works) · [Providers](#providers) · [Metrics](#metrics)

</div>

---

## Who is this for?

- **Developers without a paid subscription** who want a powerful AI coding assistant using free LLMs.
- **Developers with a paid API budget** who want to make it last — SmartSplit routes simple tasks to free models and saves your paid tokens (OpenAI, Anthropic) for complex work. No config needed, it's the default behavior.
- **Teams** who want to combine multiple LLMs without changing their existing tools.
- **Anyone** frustrated by a single model that's great at code but bad at everything else.

---

## The problem

You ask your coding assistant to write a function, explain an algorithm, translate a comment, and find the latest docs. It sends **everything to the same model** — and that model is average at most of these tasks.

**Before SmartSplit:**
```
You: "Write a Python CSV parser, explain the edge cases, and translate the docstrings to French"

→ Everything goes to one model
→ Code is okay, explanation is shallow, translation is awkward
```

**After SmartSplit:**
```
Same prompt, same client, same workflow

→ Code subtask      → best code model (deep, accurate)
→ Reasoning subtask → best reasoning model (thorough)
→ Translation       → language specialist (native quality)
→ Simple boilerplate → fast cheap model (saves your budget)
→ Combined into one coherent response
```

Same tool. Better answers. No config change.

### What makes SmartSplit different

**Multiply your free tier.** Instead of burning through one provider's quota, SmartSplit spreads requests across all your configured providers — each one contributing its free tier. More providers = more capacity.

**Self-healing.** A provider goes down or hits its rate limit? You won't even notice. SmartSplit detects failures, disables the provider temporarily, and routes to the next best one — automatically.

**Web-aware.** When your prompt needs current data ("latest", "news", "2026"...), SmartSplit detects it and searches the web before answering. No plugin needed — it's built in.

**Stretch your paid tokens.** Got an OpenAI or Anthropic API key? Add it, and SmartSplit picks the right model for each task automatically:

```
Simple task (boilerplate, summary)  → Haiku / GPT-4o-mini  (cheap)
Complex task (code, reasoning)      → Sonnet / GPT-4o      (best)
Everything else                     → Free models first
```

No config needed — SmartSplit detects task complexity and chooses the best model tier automatically.

```
Your coding assistant (Continue, Cline, Aider, Cursor...)
         |
    SmartSplit (localhost:8420)
         |
    ┌────┼──────────────────────────┐
    |    |          |                |
   Code  Search   Translate       Reasoning
    |    |          |                |
  Best   Best     Best            Best
  model  engine   model           model
    |    |          |                |
    └────┼──────────┼────────────────┘
         |
    Combined response
```

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

<details>
<summary><b>Or use Docker</b></summary>

```bash
# Create a .env file with your API keys
echo 'GROQ_API_KEY=gsk_...' > .env

# Run with Docker
docker run -p 8420:8420 --env-file .env ghcr.io/dsteinberger/smartsplit

# Or with Docker Compose
docker compose up -d
```
</details>

### 4. Connect your coding tool

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

Every request is automatically classified into one of two modes:

### RESPOND — route to the best model

Your prompt is analyzed, split into subtasks if needed, and each one is routed to the best provider:

```
"Write a Python function to parse CSV and handle errors"

  [code]       → best code model
  [reasoning]  → best reasoning model
  [synthesis]  → combines results

  → One coherent response
```

### ENRICH — search the web first, then route

When the prompt needs current data, SmartSplit searches the web first:

```
"What are the new features in Python 3.13?"

  [web_search] → search engine
  [summarize]  → best summarization model

  → Response with real, current data
```

### Context-aware

SmartSplit passes your **full conversation history** to the LLM — system prompts, previous messages, everything. For multi-subtask prompts, a context summary is injected into each subtask so no information is lost.

### Built-in reliability

| Feature | What it does |
|---------|-------------|
| **Circuit breaker** | 3 failures in 5 min → provider auto-disabled for 30 min |
| **Quality gates** | Detects refusals ("I cannot...") → auto-escalation to next provider |
| **Fallback chains** | Provider fails → next best one takes over, seamlessly |
| **Decompose cache** | Repeated prompts skip analysis (LRU, 24h TTL) |
| **Context preservation** | Full conversation history passed to each LLM |
| **Adaptive scoring** | Learns which providers work best from real results (MAB/UCB1) |

---

## Providers

### Supported providers

| Provider | Type | Best at |
|----------|------|---------|
| **Cerebras** | Free | Reasoning, general (Qwen 3 235B) |
| **Groq** | Free | Fast inference (LLaMA 3.3 70B) |
| **Gemini** | Free | Math, reasoning (Gemini 2.5 Flash) |
| **OpenRouter** | Free | Code (Qwen3 Coder 480B) |
| **Mistral** | Free | Translation (Mistral Small) |
| **HuggingFace** | Free backup | Code (Qwen2.5 Coder 32B) |
| **Cloudflare** | Free backup | General (LLaMA 3.3 70B) |
| **DeepSeek** | Paid | Code, reasoning |
| **Perplexity** | Paid | Web search + factual (Sonar/Sonar Pro) |
| **Anthropic** | Paid | Complex tasks (Claude) |
| **OpenAI** | Paid | Complex tasks (GPT-4o) |
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
export PERPLEXITY_API_KEY="pplx-..."  # paid — https://console.perplexity.ai/
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
# Using the published image
docker run -p 8420:8420 --env-file .env ghcr.io/dsteinberger/smartsplit

# Or build locally
docker build -t smartsplit .
docker run -p 8420:8420 --env-file .env smartsplit
```

Create a `.env` file with your API keys:
```bash
GROQ_API_KEY=gsk_...
SERPER_API_KEY=...
GEMINI_API_KEY=AIza...
```

> Never commit `.env` to git — it's already in `.gitignore`.

Or use Docker Compose:
```bash
docker compose up -d
```
See [`docker-compose.yml`](docker-compose.yml) for the full setup.
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
make help                 # all commands
```

> **Note:** `make test` runs all tests without any API key — no provider needed for development.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Architecture

```
smartsplit/
  proxy.py           HTTP server + LLM-based triage + CLI
  formats.py         OpenAI format conversion + SSE streaming
  planner.py         Prompt decomposition + synthesis + LRU cache
  router.py          Provider scoring + routing + quality gates
  learning.py        MAB (UCB1) adaptive scoring — learns from real results
  quota.py           Usage tracking + savings report
  config.py          Configuration + env vars
  models.py          Pydantic models + StrEnum
  exceptions.py      Custom error hierarchy
  providers/         One file per provider (3 lines for OpenAI-compatible)
```

Adding a new provider is **2 lines** (model is set in config):

```python
class NewProvider(OpenAICompatibleProvider):
    name = "new"
    api_url = "https://api.new.com/v1/chat/completions"
```

---

## Disclaimer

SmartSplit is a personal development tool. Each user must provide their own API keys and comply with the terms of service of each provider they use. SmartSplit does not store, share, or redistribute API keys or access. The authors are not responsible for any misuse or ToS violations by end users.

---

<div align="center">

MIT License · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Changelog](CHANGELOG.md)

**[Star this repo](https://github.com/dsteinberger/smartsplit)** to follow updates — new providers, streaming, and more coming soon.

</div>
