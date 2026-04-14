---
name: add-provider
description: Add a new LLM or search provider to SmartSplit. Use when the user wants to integrate a new API (e.g., Cohere, Cerebras, Brave Search).
---

# Add a New Provider to SmartSplit

Follow these steps in order:

## 1. Create the provider file

Create `smartsplit/providers/<name>.py` inheriting from `LLMProvider` (for text generation) or `SearchProvider` (for web search).

Use an existing provider as template (e.g., `groq.py` for OpenAI-compatible APIs, `gemini.py` for custom APIs).

## 2. Register in the registry

In `smartsplit/providers/registry.py`, add to `_PROVIDER_CLASSES`:
```python
"<name>": ("smartsplit.providers.<name>", "<ClassName>"),
```

If it's a free LLM, also add to `DEFAULT_FREE_LLM_PRIORITY` in `config.py`.

## 3. Add config defaults

In `smartsplit/config.py`:
- Add entry to `DEFAULT_PROVIDERS` with type, limits, model, enabled status
- Add entry to `_ENV_KEY_MAP` for the env var (e.g., `"COHERE_API_KEY": "cohere"`)
- Add scores to `DEFAULT_COMPETENCE_TABLE` for each task type

## 4. Update example config

Add the provider to `smartsplit.example.json`.

## 5. Write tests

Add tests in `tests/test_providers.py` or create a new test file. At minimum, test that the registry creates and recognizes the new provider.

## 6. Run the test suite

```bash
source .venv/bin/activate && python -m pytest tests/ -v
```

All tests must pass before considering the work done.
