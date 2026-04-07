# Contributing to SmartSplit

Thanks for your interest in SmartSplit! Here's how to get started.

## Development setup

```bash
git clone https://github.com/your-user/smartsplit.git
cd smartsplit
uv venv && uv pip install -e ".[dev]"
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

All tests must pass before submitting a PR.

## Linting and formatting

```bash
ruff check smartsplit/ tests/
ruff format smartsplit/ tests/
```

## Adding a new provider

1. Create `smartsplit/providers/<name>.py` inheriting `LLMProvider` or `SearchProvider`
2. Add to `_PROVIDER_CLASSES` in `registry.py`
3. Add defaults in `config.py` (`DEFAULT_PROVIDERS`, `_ENV_KEY_MAP`, `DEFAULT_COMPETENCE_TABLE`)
4. Add to `FREE_LLM_PRIORITY` in `registry.py` if free
5. Update `smartsplit.example.json`
6. Write tests, run the suite

See `smartsplit/providers/groq.py` for a 3-line example.

## Code style

- Python 3.11+
- `from __future__ import annotations` in every module
- Type hints on all function signatures
- No `Any`, no `# type: ignore`
- Linted with ruff

## Pull requests

- One feature per PR
- Include tests for new code
- Update docs if behavior changes
- Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`

## Reporting issues

Open an issue with:
- What you expected
- What happened
- Steps to reproduce
- Python version and OS
