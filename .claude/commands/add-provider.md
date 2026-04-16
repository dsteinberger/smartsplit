---
name: add-provider
description: Add a new LLM or search provider to SmartSplit. Use when the user wants to integrate a new API (e.g., Cohere, Perplexity, Brave Search).
---

# Add a New Provider to SmartSplit

Follow these steps in order:

## 1. Create the provider file

Create `smartsplit/providers/<name>.py`.

- **OpenAI-compatible API** (most providers): inherit `OpenAICompatibleProvider`, set `name` and `api_url`. That's it — 3 lines. See `groq.py`, `mistral.py`, `perplexity.py`.
- **Custom API** (non-OpenAI format): inherit `LLMProvider`, implement `complete()`. See `gemini.py`.
- **Search provider**: inherit `SearchProvider`, implement `search()`. See `serper.py`, `tavily.py`.

## 2. Register in the registry

In `smartsplit/providers/registry.py`, add to `_PROVIDER_CLASSES`:
```python
"<name>": ("smartsplit.providers.<name>", "<ClassName>"),
```

## 3. Add config defaults

In `smartsplit/config.py`:

### a. `DEFAULT_PROVIDERS` — add the provider entry
```python
"<name>": {
    "type": "free",          # or "paid"
    "enabled": True,         # True for free providers, False for paid (user must opt in)
    "limits": {"rpm": 100},  # rate limits if known
    "model": "model-name",   # default model
    # Paid providers only:
    "fast_model": "cheap-model",
    "strong_model": "best-model",
},
```

### b. `_ENV_KEY_MAP` — map the env var
```python
"<NAME>_API_KEY": "<name>",
```

### c. `DEFAULT_COMPETENCE_TABLE` — add scores in EVERY task category
Scores from 1-10. Free providers use bare name (`"<name>": 7`), paid providers use tiers (`"<name>:fast": 5, "<name>:strong": 8`).

Task categories: `web_search`, `summarize`, `code`, `reasoning`, `translation`, `boilerplate`, `general`, `math`, `creative`, `factual`, `extraction`.

### d. Priority lists (if applicable)
- Free LLM: add to `DEFAULT_FREE_LLM_PRIORITY`
- Potential brain: add to `_BRAIN_PRIORITY`

## 4. Write tests

Add a test in `tests/test_providers.py` that verifies the registry creates and recognizes the new provider. Run:

```bash
make check
```

All tests must pass before considering the work done.
