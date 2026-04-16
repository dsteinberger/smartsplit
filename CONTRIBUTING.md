# Contributing to SmartSplit

Thanks for your interest in SmartSplit! Here's how to get started.

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** (recommended) or pip

## Development setup

```bash
git clone https://github.com/dsteinberger/smartsplit.git
cd smartsplit
make install    # installs in editable mode with dev dependencies
```

Or manually:

```bash
# With uv
uv venv && uv pip install -e ".[dev]"

# With pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

> **No API key needed** — all tests are mocked and run offline.

## Running tests

```bash
make check    # lint + format check + tests (recommended)
make test     # tests only
make lint     # lint only
make format   # auto-format
```

All tests must pass before submitting a PR.

## Debug mode

Add `DEBUG=1` before any make command to get verbose logs (provider scores, triage decisions, predictions, worker results):

```bash
DEBUG=1 make run          # API mode
DEBUG=1 make proxy        # proxy mode
DEBUG=1 make docker-up    # Docker
```

## Adding a new provider

1. Create `smartsplit/providers/<name>.py` inheriting `LLMProvider` or `SearchProvider`
2. Add to `_PROVIDER_CLASSES` in `registry.py`
3. Add defaults in `config.py` (`DEFAULT_PROVIDERS`, `_ENV_KEY_MAP`, `DEFAULT_COMPETENCE_TABLE`)
4. Add to `DEFAULT_FREE_LLM_PRIORITY` in `config.py` if free
5. Update `smartsplit.example.json`
6. Write tests, run the suite

See `smartsplit/providers/groq.py` for a 2-line example (model is set in config).

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
